import csv
import pandas as pd

TARGET_ACTIONS = {
    'sub_car_claim_click', 'sub_car_claim_submit_click',
    'sub_open_dialog_click', 'sub_custom_question_submit_click',
    'sub_call_number_click', 'sub_callback_submit_click',
    'sub_submit_success', 'sub_car_request_submit_click'
}

def main():
    print("Читаю CSV построчно, почти без памяти...")

    agg = {}

    with open("data/ga_hits.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            session = row["session_id"]

            if session not in agg:
                agg[session] = {
                    "hits": 0,
                    "pageviews": 0,
                    "events": 0,
                    "target": 0,
                }

            agg[session]["hits"] += 1
            if row["hit_type"] == "PAGE":
                agg[session]["pageviews"] += 1
            if row["hit_type"] == "EVENT":
                agg[session]["events"] += 1
            if row["event_action"] in TARGET_ACTIONS:
                agg[session]["target"] = 1

    print("Готовлю DataFrame...")
    df = pd.DataFrame.from_dict(agg, orient="index").reset_index()
    df.rename(columns={"index": "session_id"}, inplace=True)

    df.to_pickle("data/hits_agg.pkl")
    print("Готово: hits_agg.pkl")

if __name__ == "__main__":
    main()
