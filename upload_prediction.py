"""Upload the selected agreement pages as Label Studio pre-annotations.

Set LABEL_STUDIO_API_KEY before running this entry point. It performs a dry run
by default; pass --upload to create predictions.
"""

from scripts.upload_agreement_predictions import main


if __name__ == "__main__":
    main()
