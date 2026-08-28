from scripts.qwen_exact_lift_composition_pilot import main as run_campaign


def main(argv: list[str] | None = None) -> dict:
    return run_campaign(
        argv,
        protocol="exp059-multiseed-exact-lift-composition-confirmation",
        train_seeds=(123, 456, 789),
        analysis_mode="exact_lift_multiseed",
        artifact_prefix="exp059",
        final_stem="extent-exact-lift-composition-confirmation",
        experiment_label="EXP-059 multiseed exact-lift composition confirmation",
    )


if __name__ == "__main__":
    main()
