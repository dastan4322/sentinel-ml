.PHONY: demo test lint serve

demo:
	sentinel generate --out data/synthetic.csv
	sentinel train --csv data/synthetic.csv --out artifacts/demo --source synthetic

test:
	pytest -q

lint:
	ruff check .

serve:
	uvicorn sentinel.api:app --host 127.0.0.1 --port 8000
