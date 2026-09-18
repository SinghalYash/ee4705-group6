from dataclasses import dataclass, field

from robot_skills import (
    RobotSkills,
    ActionResult,
)

# ==========================================================
# TASK 3 -> TASK 4 NAME NORMALISATION
# ==========================================================

OBJECT_ALIASES = {

    # Box
    "box": "box",
    "blue box": "box",
    "blue_box": "box",
    "cube": "box",
    "blue cube": "box",

    # Cylinder
    "cylinder": "cylinder",
    "green cylinder": "cylinder",
    "green_cylinder": "cylinder",

    # Stone
    "stone": "stone",
    "rock": "stone",
    "grey stone": "stone",
    "gray stone": "stone",
    "sphere": "stone",
}

TARGET_ALIASES = {

    "red_area": "red_area",
    "red area": "red_area",
    "red zone": "red_area",
    "red region": "red_area",
    "red target": "red_area",
    "target area": "red_area",
}

# ==========================================================
# EXECUTION REPORT
# ==========================================================

@dataclass
class ExecutionReport:
    """
    Summary of one complete structured plan execution.
    """

    success: bool

    completed_actions: int

    total_actions: int

    results: list = field(
        default_factory=list
    )

    failure_reason: str | None = None

    final_placement_error: float | None = None

    grasp_attempts: int = 0


# ==========================================================
# TASK 4 EXECUTOR
# ==========================================================

class Task4Executor:

    def __init__(
        self,
        robot=None,
    ):

        if robot is None:

            self.robot = (
                RobotSkills()
            )

        else:

            self.robot = robot

        # --------------------------------------------------
        # Execution context
        # --------------------------------------------------

        self.last_grasped_object = None

        self.last_target_region = None

        self.last_placed_object = None

        self.execution_results = []

    # ======================================================
    # NAME NORMALISATION
    # ======================================================

    def normalise_object_name(
        self,
        name,
    ):

        if name is None:
            return None

        key = str(
            name
        ).strip().lower()

        return OBJECT_ALIASES.get(
            key,
            key,
        )


    def normalise_target_name(
        self,
        name,
    ):

        if name is None:
            return None

        key = str(
            name
        ).strip().lower()

        return TARGET_ALIASES.get(
            key,
            key,
        )

    # ======================================================
    # START
    # ======================================================

    def start(self):

        if self.robot.viewer is None:

            self.robot.start_viewer()

    # ======================================================
    # ACTION DISPATCH
    # ======================================================

    def execute_action(
        self,
        action,
    ):
        """
        Convert one structured Task 3 action into a
        physical Task 4 robot behaviour.
        """

        if not isinstance(
            action,
            dict,
        ):

            return ActionResult(
                success=False,
                skill="INVALID",
                message=(
                    "Action must be "
                    "a dictionary"
                ),
            )

        skill = action.get(
            "skill"
        )

        if skill is None:

            return ActionResult(
                success=False,
                skill="INVALID",
                message=(
                    "Action has no skill field"
                ),
            )

        skill = str(
            skill
        ).upper()

        print(
            "\n"
            + "#" * 60
        )

        print(
            f"TASK 4 EXECUTOR: {skill}"
        )

        print(
            "Action:",
            action,
        )

        print(
            "#" * 60
        )

        # ==================================================
        # APPROACH
        # ==================================================

        if skill == "APPROACH":

            target = (
                self.normalise_object_name(
                    action.get(
                        "target"
                    )
                )
            )

            if not target:

                return ActionResult(
                    success=False,
                    skill="APPROACH",
                    message=(
                        "APPROACH requires "
                        "a target"
                    ),
                )

            return self.robot.approach(
                target
            )

        # ==================================================
        # REACH
        # ==================================================

        if skill == "REACH":

            target = action.get(
                "target"
            )

            if not target:

                return ActionResult(
                    success=False,
                    skill="REACH",
                    message=(
                        "REACH requires "
                        "a target"
                    ),
                )

            return self.robot.reach(
                target
            )

        # ==================================================
        # GRASP
        # ==================================================

        if skill == "GRASP":

            target = (
                self.normalise_object_name(
                    action.get(
                        "target"
                    )
                )
            )

            if not target:

                return ActionResult(
                    success=False,
                    skill="GRASP",
                    message=(
                        "GRASP requires "
                        "a target"
                    ),
                )

            # ------------------------------------------
            # Ensure pre-grasp state
            # ------------------------------------------
            #
            # Task 3 may generate:
            #
            # APPROACH -> GRASP
            #
            # without an explicit REACH action.
            #
            # Task 4 therefore ensures that the physical
            # precondition for GRASP is satisfied.

            if (
                self.robot.current_object
                != target
            ):

                print(
                    "\nGRASP precondition "
                    "not yet satisfied."
                )

                print(
                    "Executing REACH "
                    "automatically..."
                )

                reach_result = (
                    self.robot.reach(
                        target
                    )
                )

                if not reach_result.success:

                    return ActionResult(
                        success=False,
                        skill="GRASP",
                        message=(
                            "Could not reach "
                            f"{target} before "
                            "grasp"
                        ),
                        error=(
                            reach_result.error
                        ),
                    )

            # ------------------------------------------
            # Recovery-enabled grasp
            # ------------------------------------------

            result = (
                self.robot.grasp_with_recovery(
                    target
                )
            )

            if result.success:

                self.last_grasped_object = (
                    target
                )

            return result

        # ==================================================
        # MOVE_TO
        # ==================================================

        if skill == "MOVE_TO":

            target = (
                self.normalise_target_name(
                    action.get(
                        "target"
                    )
                )
            )

            if not target:

                return ActionResult(
                    success=False,
                    skill="MOVE_TO",
                    message=(
                        "MOVE_TO requires "
                        "a target"
                    ),
                )

            result = self.robot.move_to(
                target
            )

            if result.success:

                self.last_target_region = (
                    target
                )

            return result

        # ==================================================
        # PLACE
        # ==================================================

        if skill == "PLACE":

            object_name = (
                self.normalise_object_name(
                    action.get(
                        "object"
                    )
                )
            )

            target = (
                self.normalise_target_name(
                    action.get(
                        "target"
                    )
                )
            )

            if not object_name:

                object_name = (
                    self.last_grasped_object
                )

            if not target:

                target = (
                    self.last_target_region
                )

            result = self.robot.place(
                object_name,
                target,
            )

            if result.success:

                self.last_placed_object = (
                    object_name
                )

                self.last_target_region = (
                    target
                )

            return result

        # ==================================================
        # VERIFY
        # ==================================================

        if skill == "VERIFY":

            return self.execute_verify(
                action
            )

        # ==================================================
        # STOP
        # ==================================================

        if skill == "STOP":

            reason = action.get(
                "reason",
                "task complete",
            )

            return self.robot.stop(
                reason
            )


        # ==================================================
        # SEARCH
        # ==================================================

                # ==================================================
        # SEARCH
        # ==================================================

        if skill == "SEARCH":

            target = (
                self.normalise_object_name(
                    action.get(
                        "target"
                    )
                )
            )

            if not target:

                return ActionResult(
                    success=False,
                    skill="SEARCH",
                    message=(
                        "SEARCH requires "
                        "a target"
                    ),
                )

            return self.robot.search(
                target
            )
        # ==================================================
        # UNKNOWN SKILL
        # ==================================================

        return ActionResult(
            success=False,
            skill=skill,
            message=(
                f"Unsupported Task 4 "
                f"skill: {skill}"
            ),
        )

    # ======================================================
    # VERIFY DISPATCH
    # ======================================================

    def execute_verify(
        self,
        action,
    ):
        """
        Interpret Task 3 VERIFY actions.

        Current supported checks:

        1. grasp verification
        2. placement verification

        Later Task 2 can provide visual verification
        through this same interface.
        """

        condition = str(
            action.get(
                "condition",
                "",
            )
        ).lower()

        object_name = action.get(
            "object"
        )

        target = action.get(
            "target"
        )

        # --------------------------------------------------
        # Explicit structured grasp verification
        # --------------------------------------------------

        verify_type = str(
            action.get(
                "verify_type",
                "",
            )
        ).upper()

        if (
            verify_type
            == "GRASP"
        ):

            if not object_name:

                object_name = (
                    self.last_grasped_object
                )

            if not object_name:

                return ActionResult(
                    success=False,
                    skill="VERIFY",
                    message=(
                        "No object available "
                        "for grasp verification"
                    ),
                )

            return (
                self.robot.verify_grasp(
                    object_name
                )
            )

        # --------------------------------------------------
        # Explicit structured placement verification
        # --------------------------------------------------

        if (
            verify_type
            == "PLACE"
        ):

            if not object_name:

                object_name = (
                    self.last_placed_object
                )

            if not target:

                target = (
                    self.last_target_region
                )

            if (
                not object_name
                or not target
            ):

                return ActionResult(
                    success=False,
                    skill="VERIFY",
                    message=(
                        "Missing object or "
                        "target for placement "
                        "verification"
                    ),
                )

            return (
                self.robot.verify_place(
                    object_name,
                    target,
                )
            )

        # --------------------------------------------------
        # Compatibility with Task 3 condition strings
        # --------------------------------------------------

        if (
            "grasp" in condition
            or "holding" in condition
            or "held" in condition
        ):

            if not object_name:

                object_name = (
                    self.last_grasped_object
                )

            if not object_name:

                return ActionResult(
                    success=False,
                    skill="VERIFY",
                    message=(
                        "Could not determine "
                        "object for grasp "
                        "verification"
                    ),
                )

            return (
                self.robot.verify_grasp(
                    object_name
                )
            )

        # Placement conditions such as:
        #
        # "box in red_area"
        # "box is in red_area"
        # "object placed in target"
        if (
            "red_area" in condition
            or "placed" in condition
            or "target" in condition
            or " in " in condition
        ):

            if not object_name:

                object_name = (
                    self.last_placed_object
                )

            if not target:

                target = (
                    self.last_target_region
                )

            if (
                not object_name
                or not target
            ):

                return ActionResult(
                    success=False,
                    skill="VERIFY",
                    message=(
                        "Could not determine "
                        "object/target for "
                        "placement verification"
                    ),
                )

            return (
                self.robot.verify_place(
                    object_name,
                    target,
                )
            )

        # --------------------------------------------------
        # Context fallback
        # --------------------------------------------------

        if (
            self.last_placed_object
            is not None
            and self.last_target_region
            is not None
        ):

            return (
                self.robot.verify_place(
                    self.last_placed_object,
                    self.last_target_region,
                )
            )

        if (
            self.last_grasped_object
            is not None
        ):

            return (
                self.robot.verify_grasp(
                    self.last_grasped_object
                )
            )

        return ActionResult(
            success=False,
            skill="VERIFY",
            message=(
                "VERIFY condition could "
                "not be interpreted"
            ),
        )

    # ======================================================
    # COMPLETE PLAN EXECUTION
    # ======================================================

    def execute_plan(
        self,
        plan,
    ):
        """
        Execute an entire structured Task 3 plan.

        Execution stops immediately when an action fails.
        """

        print(
            "\n"
            + "=" * 65
        )

        print(
            "TASK 4 — STRUCTURED PLAN EXECUTION"
        )

        print(
            "=" * 65
        )

        # --------------------------------------------------
        # Validate plan
        # --------------------------------------------------

        if not isinstance(
            plan,
            dict,
        ):

            return ExecutionReport(
                success=False,
                completed_actions=0,
                total_actions=0,
                failure_reason=(
                    "Plan must be "
                    "a dictionary"
                ),
            )

        feasible = plan.get(
            "feasible",
            True,
        )

        if not feasible:

            reason = plan.get(
                "reason",
                "Task 3 marked plan "
                "as infeasible",
            )

            self.robot.safe_stop(
                reason
            )

            return ExecutionReport(
                success=False,
                completed_actions=0,
                total_actions=0,
                failure_reason=reason,
            )

        actions = plan.get(
            "actions",
            []
        )

        if not actions:

            reason = (
                "Structured plan "
                "contains no actions"
            )

            self.robot.safe_stop(
                reason
            )

            return ExecutionReport(
                success=False,
                completed_actions=0,
                total_actions=0,
                failure_reason=reason,
            )

        # --------------------------------------------------
        # Start simulator
        # --------------------------------------------------

        self.start()

        self.execution_results = []

        completed_actions = 0

        # --------------------------------------------------
        # Execute sequentially
        # --------------------------------------------------

        for index, action in enumerate(
            actions,
            start=1,
        ):

            print(
                "\n"
                + "=" * 65
            )

            print(
                f"ACTION {index}/"
                f"{len(actions)}"
            )

            print(
                "=" * 65
            )

            result = (
                self.execute_action(
                    action
                )
            )

            self.execution_results.append(
                result
            )

            print(
                "\n--- ACTION RESULT ---"
            )

            print(
                "Success:",
                result.success,
            )

            print(
                "Skill:",
                result.skill,
            )

            print(
                "Message:",
                result.message,
            )

            print(
                "Error:",
                result.error,
            )

            # ------------------------------------------
            # Failure
            # ------------------------------------------

            if not result.success:

                failure_reason = (
                    result.message
                )

                # GRASP recovery may already have
                # triggered safe_stop().
                if not self.robot.safe_stopped:

                    self.robot.safe_stop(
                        (
                            f"{result.skill} "
                            f"failed: "
                            f"{failure_reason}"
                        )
                    )

                return ExecutionReport(
                    success=False,
                    completed_actions=(
                        completed_actions
                    ),
                    total_actions=len(
                        actions
                    ),
                    results=(
                        self.execution_results
                    ),
                    failure_reason=(
                        failure_reason
                    ),
                    final_placement_error=(
                        self._get_final_place_error()
                    ),
                    grasp_attempts=(
                        self.robot.grasp_attempts
                    ),
                )

            completed_actions += 1

        # --------------------------------------------------
        # Entire plan succeeded
        # --------------------------------------------------

        print(
            "\n"
            + "=" * 65
        )

        print(
            "TASK 4 PLAN EXECUTION SUCCESSFUL"
        )

        print(
            "=" * 65
        )

        return ExecutionReport(
            success=True,
            completed_actions=(
                completed_actions
            ),
            total_actions=len(
                actions
            ),
            results=(
                self.execution_results
            ),
            failure_reason=None,
            final_placement_error=(
                self._get_final_place_error()
            ),
            grasp_attempts=(
                self.robot.grasp_attempts
            ),
        )

    # ======================================================
    # RESULT HELPER
    # ======================================================

    def _get_final_place_error(
        self,
    ):

        for result in reversed(
            self.execution_results
        ):

            if (
                result.skill
                in [
                    "PLACE",
                    "VERIFY",
                ]
                and result.error
                is not None
            ):

                return float(
                    result.error
                )

        return None