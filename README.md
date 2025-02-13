# DATA511: FPL Dashboard

## Deployed Dashboard

Follow this link to try it out! https://fpl511.streamlit.app/

## Execution

Run the following within your conda env to execute the main dash:

```bash
streamlit run app.py
```

Navigate to:

```bash
http://localhost:8501/
```

The app selects port 8501 by default, but can be controlled by including the flag: --server.port << NEW PORT NUMBER HERE >>

Make sure you are forwarding the correct ports if running remotely

## Config

Open:
```
vim ~/.streamlit/config.toml
```

Add the following line:
```
[theme]
base="dark"
```
