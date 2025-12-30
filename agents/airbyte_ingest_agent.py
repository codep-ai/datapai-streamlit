# agents/airbyte_ingest_agent.py

import os
import time
import requests


def ingest_with_airbyte(connection_id: str, poll: bool = True) -> tuple[bool, str]:
    """
    Trigger an Airbyte sync by connection ID.
    Optionally poll the job until completion.
    """
    base_url = os.environ.get("AIRBYTE_API_URL", "http://localhost:8000/api")

    # Trigger sync
    trigger_url = f"{base_url}/v1/connections/sync"
    response = requests.post(trigger_url, json={"connectionId": connection_id})

    if response.status_code != 200:
        return False, f"Failed to trigger Airbyte sync: {response.text}"

    job_id = response.json()["job"]["id"]
    print(f"✅ Triggered Airbyte sync: job ID {job_id}")

    if not poll:
        return True, f"Sync job triggered: {job_id}"

    # Poll until job completes
    status_url = f"{base_url}/v1/jobs/get"
    while True:
        res = requests.post(status_url, json={"id": job_id})
        if res.status_code != 200:
            return False, f"Error polling job: {res.text}"
        job = res.json()["job"]
        status = job["status"]
        print(f"Airbyte job status: {status}")
        if status in ("succeeded", "failed", "cancelled", "incomplete"):
            break
        time.sleep(10)

    return (status == "succeeded", f"Airbyte job {status}")

