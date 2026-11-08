from __future__ import annotations

from pathlib import Path
import json
import torch

from reliable_face_defense_agents.red.agent import RedAgent, RedAgentConfig

from .data import load_rgb01
from .reporting import save_image01, save_perturbation, json_dump, write_csv
from .red_eval import ContextualRedViolationEvaluator
from reliable_face_defense_rpp.metrics import mae, psnr, ssim
from .deployability import (
    FinalDeploymentGate,
    VisualGateConfig,
)


class FullSystemOrchestrator:
    """
    Integrated method loop:

      Blue verified dynamic budget/route
        -> hard reliability
        -> OR-correct RPP
        -> Red control-set worst propagation search
        -> EMA + Top-K + monotonic active constraints
        -> next Blue round

    After Red rounds stabilize/end, the final active set is frozen and used on
    the test split.
    """
    def __init__(
        self,
        *,
        cfg,
        reliability,
        blue,
        rpp_factory,
        hard_propagation,
        red_transforms,
        output_dir,
        deployment_gate=None,
    ):
        self.cfg=cfg
        self.reliability=reliability
        self.blue=blue
        self.rpp=rpp_factory

        # Backward-compatible default for tests / older callers.
        # No visual threshold is invented here.
        if deployment_gate is None:
            deployment_gate = FinalDeploymentGate(
                VisualGateConfig(
                    enabled=False,
                    min_psnr_db=None,
                    max_lpips=None,
                )
            )

        self.deployment_gate=deployment_gate
        self.hard=hard_propagation
        self.red_transforms=tuple(red_transforms)
        self.output=Path(output_dir)
        self.output.mkdir(parents=True,exist_ok=True)

        self.round_index=0
        self.red=None

    def _prepare_split(self, paths):
        images={}
        contexts={}
        for p in paths:
            sid=Path(p).stem
            x=load_rgb01(p,self.cfg.image_size).unsqueeze(0).to(self.cfg.device)
            images[sid]=(p,x)
            contexts[sid]=self.reliability.prepare_context(sid,x)
        return images,contexts

    def _make_red(self,contexts):
        evaluator=ContextualRedViolationEvaluator(
            reliability=self.reliability,
            contexts=contexts,
        )
        return RedAgent(
            evaluator=evaluator,
            transforms=self.red_transforms,
            config=RedAgentConfig(
                ema_alpha=self.cfg.red.ema_alpha,
                top_k=self.cfg.red.top_k,
            ),
        )

    def _process_one(
        self,
        *,
        sample_id,
        path,
        clean,
        context,
        active_constraints,
        round_dir,
    ):
        decision=self.blue.act(
            sample_id=sample_id,
            clean01=clean,
            context=context,
            active_constraints=active_constraints,
        )
        sample_dir=round_dir/sample_id
        sample_dir.mkdir(parents=True,exist_ok=True)
        save_image01(clean,sample_dir/"source.png")
        json_dump(sample_dir/"blue.json",decision.to_dict())

        if not decision.feasible:
            return {
                "sample_id":sample_id,
                "status":"INFEASIBLE",
                "protected":None,
                "blue":decision,
                "rpp":None,
            }

        save_image01(decision.chosen.protected,sample_dir/"blue_protected.png")
        eps_vis=decision.chosen.epsilon_px or (
            float((decision.chosen.protected-clean).abs().max().detach().cpu())*255
        )
        save_perturbation(
            clean,decision.chosen.protected,
            sample_dir/"blue_perturbation_vis.png",
            scale=max(eps_vis,1e-6),
        )

        rpp=self.rpp.refine(
            clean01=clean,
            blue_candidate=decision.chosen,
            context=context,
            reliability=self.reliability,
            hard_propagation=self.hard,
        )
        save_image01(rpp.refined,sample_dir/"protected.png")
        save_perturbation(
            clean,rpp.refined,
            sample_dir/"perturbation_vis.png",
            scale=max(eps_vis,1e-6),
        )
        json_dump(
            sample_dir/"rpp.json",
            {
                "status":rpp.status,
                "certificate":rpp.certificate.branch_by_environment,
                "accepted_steps":rpp.accepted_steps,
                "rollback_steps":rpp.rollback_steps,
                "total_projections":rpp.total_projections,
                "final_task_reliable":rpp.final_snapshot.task_reliable,
                "final_certificate_reliable":rpp.final_snapshot.certificate_reliable,
            }
        )
        if not rpp.final_snapshot.certificate_reliable:
            raise RuntimeError("RPP returned a non-reliable protected image.")

        return {
            "sample_id":sample_id,
            "status":"OK",
            "protected":rpp.refined,
            "blue":decision,
            "rpp":rpp,
        }

    def _robustness_sweep(self, protected, context):
        """
        Final evaluation only: evaluate CLEAN + every transform in the selected
        Red pool. This does not add new training constraints and therefore
        cannot leak final-test information back into Blue/Red training.
        """
        per={}
        worst_env=None
        worst_q=-1.0
        all_reliable=True

        clean_ev=self.reliability.evaluate(protected,context)
        clean_q=float(clean_ev.v_detection+clean_ev.v_identity)
        per["clean"]={**clean_ev.to_dict(),"q":clean_q}
        all_reliable=all_reliable and clean_ev.defense_success
        worst_env="clean"
        worst_q=clean_q

        for spec in self.red_transforms:
            y=self.hard.apply(spec.name,protected)
            ev=self.reliability.evaluate(y,context)
            q=float(ev.v_detection+ev.v_identity)
            per[spec.name]={**ev.to_dict(),"q":q}
            all_reliable=all_reliable and ev.defense_success
            if q>worst_q:
                worst_q=q
                worst_env=spec.name

        return {
            "robust_all_pool_reliable":bool(all_reliable),
            "worst_environment":worst_env,
            "worst_q":float(worst_q),
            "per_environment":per,
        }

    def run_control_rounds(self,control_paths):
        images,contexts=self._prepare_split(control_paths)
        self.red=self._make_red(contexts)

        history=[]
        previous_active=()
        start_round=0
        state_path=self.output/"orchestrator_state.pt"
        if state_path.exists():
            state=torch.load(state_path,map_location="cpu",weights_only=False)
            self.red.load_state_dict(state["red"])
            start_round=int(state["next_round"])
            previous_active=self.red.active_constraints

        for round_idx in range(start_round,self.cfg.red.max_rounds):
            self.round_index=round_idx
            active=self.red.active_constraints
            round_dir=self.output/f"round_{round_idx:02d}"
            protected_control=[]
            records=[]

            for sid,(path,clean) in images.items():
                rec=self._process_one(
                    sample_id=sid,
                    path=path,
                    clean=clean,
                    context=contexts[sid],
                    active_constraints=active,
                    round_dir=round_dir,
                )
                records.append(rec)
                if rec["protected"] is not None:
                    protected_control.append((sid,rec["protected"]))

            if not protected_control:
                raise RuntimeError(
                    "All control samples are Blue-infeasible; Red cannot update."
                )

            red_decision=self.red.search(
                control_samples=protected_control,
            )
            json_dump(
                round_dir/"red.json",
                red_decision.to_dict(),
            )
            history.append({
                "round":round_idx,
                "active_before":list(active),
                "red_topk":list(red_decision.selected_topk),
                "active_after":list(red_decision.active_constraints),
                "feasible_samples":len(protected_control),
                "total_samples":len(images),
            })

            new_active=red_decision.active_constraints
            stable=(new_active==previous_active) or (new_active==active)
            previous_active=new_active

            self.save_state(round_idx+1)
            if self.cfg.red.stop_when_no_new_constraint and stable:
                break

        json_dump(self.output/"control_history.json",history)
        write_csv(
            self.output/"control_history.csv",
            history,
            fieldnames=[
                "round","active_before","red_topk","active_after",
                "feasible_samples","total_samples",
            ],
        )
        return tuple(self.red.active_constraints),history

    def run_test(self,test_paths,active_constraints):
        images,contexts=self._prepare_split(test_paths)
        out_dir=self.output/"final_test"
        rows=[]
        for sid,(path,clean) in images.items():
            rec=self._process_one(
                sample_id=sid,
                path=path,
                clean=clean,
                context=contexts[sid],
                active_constraints=active_constraints,
                round_dir=out_dir,
            )
            row={
                "sample_id":sid,
                "source_path":path,
                "status":rec["status"],
                "active_constraints":list(active_constraints),
                "_sample_dir":str(out_dir/sid),
            }
            if rec["blue"].feasible:
                sweep=self._robustness_sweep(
                    rec["rpp"].refined,
                    contexts[sid],
                )
                final_mae = mae(
                    clean,
                    rec["rpp"].refined,
                )
                final_psnr = psnr(
                    clean,
                    rec["rpp"].refined,
                )
                final_ssim = ssim(
                    clean,
                    rec["rpp"].refined,
                )
                final_lpips = self.blue.distortion(
                    clean,
                    rec["rpp"].refined,
                )

                # Security reliability for the actual active constraint set
                # used by Blue/RPP.
                security_reliable = bool(
                    rec["rpp"].final_snapshot.task_reliable
                    and
                    rec["rpp"].final_snapshot.certificate_reliable
                )

                deployment = self.deployment_gate.evaluate(
                    security_reliable=security_reliable,
                    psnr_db=final_psnr,
                    lpips=final_lpips,
                )

                row.update({
                    "route":rec["blue"].chosen.route.value,
                    "epsilon_px":rec["blue"].chosen.epsilon_px,

                    "distortion_lpips":
                        rec["blue"].chosen.distortion,

                    "searched_facepoison_eps":list(
                        rec["blue"].searched_facepoison_eps
                    ),

                    "rpp_status":rec["rpp"].status,
                    "rpp_accepted_steps":
                        rec["rpp"].accepted_steps,
                    "rpp_rollbacks":
                        rec["rpp"].rollback_steps,

                    # Legacy field retained.
                    "final_task_reliable":
                        rec["rpp"].final_snapshot.task_reliable,

                    # Explicit new semantics.
                    "security_reliable":
                        security_reliable,

                    "final_mae":final_mae,
                    "final_psnr":final_psnr,
                    "final_ssim":final_ssim,
                    "final_lpips":final_lpips,

                    "visual_gate_calibrated":
                        deployment.visual_gate_calibrated,
                    "visual_psnr_ok":
                        deployment.psnr_ok,
                    "visual_lpips_ok":
                        deployment.lpips_ok,
                    "visual_acceptable":
                        deployment.visual_acceptable,
                    "deployable":
                        deployment.deployable,

                    "final_active_per_environment":
                        rec["rpp"].final_snapshot.per_environment,

                    **sweep,
                })

                json_dump(
                    out_dir/sid/"deployment.json",
                    deployment.to_dict(),
                )
                json_dump(
                    out_dir/sid/"robustness_sweep.json",
                    sweep,
                )
            rows.append(row)
        json_dump(out_dir/"results.json",rows)
        flat=[]
        for row in rows:
            flat.append({
                "sample_id":row.get("sample_id"),
                "source_path":row.get("source_path"),
                "status":row.get("status"),
                "route":row.get("route"),
                "epsilon_px":row.get("epsilon_px"),
                "distortion_lpips":row.get("distortion_lpips"),
                "rpp_status":row.get("rpp_status"),
                "rpp_accepted_steps":row.get("rpp_accepted_steps"),
                "rpp_rollbacks":row.get("rpp_rollbacks"),
                "final_task_reliable":row.get("final_task_reliable"),
                "security_reliable":row.get("security_reliable"),
                "visual_gate_calibrated":row.get("visual_gate_calibrated"),
                "visual_psnr_ok":row.get("visual_psnr_ok"),
                "visual_lpips_ok":row.get("visual_lpips_ok"),
                "visual_acceptable":row.get("visual_acceptable"),
                "deployable":row.get("deployable"),
                "robust_all_pool_reliable":row.get("robust_all_pool_reliable"),
                "worst_environment":row.get("worst_environment"),
                "worst_q":row.get("worst_q"),
                "final_mae":row.get("final_mae"),
                "final_psnr":row.get("final_psnr"),
                "final_ssim":row.get("final_ssim"),
                "final_lpips":row.get("final_lpips"),
            })
        write_csv(
            out_dir/"results.csv",
            flat,
            fieldnames=[
                "sample_id","source_path","status","route","epsilon_px",
                "distortion_lpips","rpp_status","rpp_accepted_steps",
                "rpp_rollbacks","final_task_reliable",
                "security_reliable","visual_gate_calibrated",
                "visual_psnr_ok","visual_lpips_ok",
                "visual_acceptable","deployable",
                "robust_all_pool_reliable","worst_environment","worst_q",
                "final_mae","final_psnr","final_ssim","final_lpips",
            ],
        )
        return rows

    def save_state(self,next_round):
        if self.red is None:
            return
        torch.save({
            "next_round":int(next_round),
            "red":self.red.state_dict(),
        },self.output/"orchestrator_state.pt")
