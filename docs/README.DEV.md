
## Development Usage

* Step1:Start Environment.

    ```bash
    #Interactive mode
    docker compose -f ./compose.dev.yaml run --rm core

    ```
* Step2: Start service.
    ```bash
    #Interactive mode
    python3 app.py

    # Operate With WebUI :http://127.0.0.1:8000/static/index.html
    # Open api docs : http://127.0.0.1:8000/docs

    ```
* Finally: Shutdown service
    ```bash
    docker compose down
    ```

## Other
* [Update map](/docs/UPDATE.md)
* [Todo](/docs/TODO.md)
