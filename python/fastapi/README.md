# fastapi

1. link
   * [github/fastapi](https://github.com/tiangolo/fastapi)
   * [fastapi/documentation](https://fastapi.tiangolo.com/)
   * [github/openapi](https://github.com/OAI/OpenAPI-Specification)
   * [cryptography/github](https://github.com/pyca/cryptography) [cryptography/documentation](https://cryptography.io/en/latest/)
   * [jose/github](https://github.com/mpdavis/python-jose) [jose/documentation](https://python-jose.readthedocs.org/en/latest/)
   * [passlib/repo](https://foss.heptapod.net/python-libs/passlib) [passlib/documentation](https://passlib.readthedocs.io/en/stable/) TODO replaced with `cryptography.hazmat.primitives.kdf.scrypt`
2. install
   * `pip install fastapi "uvicorn[standard]" python-multipart python-jose[cryptography] cryptography passlib`
   * `conda install -c conda-forge uvicorn fastapi python-multipart python-jose cryptography passlib`
3. preference
   * **forbid** using query parameter of boolean type, `1`, `true`, `on`, `yes`, `True`
4. concept
   * data schema, json schema
   * api schema, openapi `http://127.0.0.1:8000/openapi.json`
   * query parameter `?skip=0&limit=10`
   * JWT: Json Web Token [jwt.io](https://jwt.io/)
   * JOSE: JavaScript Object Signing and Encryption
   * JWS: JSON Web Signature
5. security
   * OAuth2
   * OpenID connect
   * OpenAPI: apikey, http, oauth2, openIdConnect

```bash
uvicorn draft00:app --reload
# default port: --port 8000
# http://localhost:8000/doc
# http://localhost:8000/redoc
```
