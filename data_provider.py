"""Read-only football-data.org v4 integration."""
import requests

BASE = "https://api.football-data.org/v4"

class FootballAPIError(RuntimeError):
    pass


def get_json(path: str, token: str, params: dict | None = None) -> dict:
    if not token or not token.strip():
        raise FootballAPIError("Χρειάζεται API token.")
    try:
        response = requests.get(BASE + path, params=params, headers={"X-Auth-Token": token.strip()}, timeout=18)
    except requests.RequestException as exc:
        raise FootballAPIError(f"Πρόβλημα σύνδεσης με το API: {exc}") from exc
    if response.status_code != 200:
        try:
            detail = (response.json() or {}).get("message") or (response.json() or {}).get("error", "")
        except ValueError:
            detail = ""
        if response.status_code == 429:
            raise FootballAPIError("Υπέρβαση API rate limit (429). Περίμενε πριν ξαναφορτώσεις.")
        if response.status_code in (401, 403):
            raise FootballAPIError(f"Κωδικός {response.status_code}: έλεγξε token ή συνδρομητική πρόσβαση. {detail}")
        raise FootballAPIError(f"HTTP {response.status_code}: {detail}")
    return response.json()


def competitions(token: str) -> list[dict]:
    return (get_json("/competitions", token).get("competitions") or [])


def competition_matches(token: str, code: str) -> list[dict]:
    return get_json(f"/competitions/{code}/matches", token).get("matches") or []
