from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


CATEGORY_CONTENT = {
    "billing": {
        "subjects": [
            "Unexpected charge on my account",
            "Payment was charged twice",
            "Invoice amount looks incorrect",
            "Card payment is not going through",
            "Subscription billing question",
        ],
        "details": [
            "The amount on my card statement does not match the invoice.",
            "I can see two charges for the same order in my bank account.",
            "The payment page shows an error after I submit my card details.",
            "Please explain the charge and help me understand the invoice.",
            "My subscription renewed and I need someone to check the billing record.",
        ],
    },
    "delivery": {
        "subjects": [
            "My order has not arrived",
            "Package tracking has not updated",
            "Delivery address needs correction",
            "The parcel arrived damaged",
            "Courier marked my order delivered",
        ],
        "details": [
            "The tracking page has shown the same location for several days.",
            "The courier says delivered but I have not received the package.",
            "The parcel arrived with damaged packaging and missing items.",
            "I need help checking the expected delivery date for my order.",
            "I entered the wrong apartment number and need to update the address.",
        ],
    },
    "account": {
        "subjects": [
            "Unable to sign in to my account",
            "Password reset link is not working",
            "My account was locked",
            "Need to update account details",
            "Verification code never arrived",
        ],
        "details": [
            "The sign-in page keeps rejecting my correct password.",
            "The password reset link opens an expired page.",
            "I cannot receive the one-time verification code on my phone.",
            "My profile details are incorrect and I cannot edit them.",
            "The system says my account is temporarily locked.",
        ],
    },
    "technical": {
        "subjects": [
            "Application crashes when I open it",
            "Website page is not loading",
            "Unable to upload a document",
            "Search function returns an error",
            "Dashboard is showing a blank screen",
        ],
        "details": [
            "The application closes immediately after I open the dashboard.",
            "I receive a server error whenever I try to load this page.",
            "The upload stops halfway and displays an unexpected error.",
            "The search feature is not returning any results for valid terms.",
            "The page worked yesterday but now the interface is blank.",
        ],
    },
    "refund": {
        "subjects": [
            "Requesting a refund for my order",
            "Refund has not reached my bank account",
            "Order was cancelled but money was not returned",
            "I received the wrong item and need a refund",
            "Refund status is still pending",
        ],
        "details": [
            "I returned the item and want to know when the refund will arrive.",
            "The order was cancelled but the original payment is still visible.",
            "I received a different product and would prefer my money back.",
            "The refund status has not changed since the request was submitted.",
            "Please check whether the refund was processed successfully.",
        ],
    },
}

PRIORITY_CONTEXT = {
    "low": [
        "This is not urgent and can be reviewed whenever the team is available.",
        "There is no immediate impact; I am asking for general guidance.",
        "I can continue using the service while this is being checked.",
    ],
    "medium": [
        "This is affecting one task and I would appreciate an update soon.",
        "The issue is inconvenient but I have a temporary workaround.",
        "Please review this during the normal support process.",
    ],
    "high": [
        "This is blocking an important task and needs prompt attention.",
        "Several team members are affected and our work is delayed.",
        "The issue is causing a significant disruption to our workflow.",
    ],
    "critical": [
        "Our production service is down and all users are affected; this is urgent.",
        "This is a business-critical outage with no available workaround; act immediately.",
        "All transactions are failing and the issue requires emergency attention now.",
    ],
}



def build_dataset(samples_per_pair: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for category, content in CATEGORY_CONTENT.items():
        for priority, priority_phrases in PRIORITY_CONTEXT.items():
            for _ in range(samples_per_pair):
                subject = rng.choice(content["subjects"])
                detail = rng.choice(content["details"])
                severity = rng.choice(priority_phrases)
                variation = rng.choice(
                    [
                        "Please let me know the next steps.",
                        "I have included the details so the team can investigate.",
                        "Thank you for looking into this request.",
                        "Could you check this and share an update?",
                        "I can provide more information if required.",
                    ]
                )
                rows.append(
                    {
                        "subject": subject,
                        "description": f"{detail} {severity} {variation}",
                        "category": category,
                        "priority": priority,
                    }
                )
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clearly synthetic SupportIQ demo dataset.")
    parser.add_argument("--samples-per-pair", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "sample_tickets.csv",
    )
    args = parser.parse_args()
    if args.samples_per_pair < 5:
        parser.error("--samples-per-pair must be at least 5 so the train/test split is meaningful.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(args.samples_per_pair, args.seed)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset)} synthetic rows to {args.output}")
    print("Categories:", sorted(dataset["category"].unique().tolist()))
    print("Priorities:", sorted(dataset["priority"].unique().tolist()))
    print("IMPORTANT: this generated dataset is for pipeline testing, not real-world model claims.")


if __name__ == "__main__":
    main()
