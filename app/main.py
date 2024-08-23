import os
from fastapi import FastAPI, HTTPException, Header, Depends
from app.routers import router
# from typing_extensions import Annotated

# SECRET_TOKEN = os.getenv("SECRET_TOKEN")

# async def verify_token(x_token: Annotated[str, Header()]) -> None:
#     """Verify the token is valid."""

#     if x_token != SECRET_TOKEN:
#         raise HTTPException(status_code=400, detail="X-Token header invalid")

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="API Server de prueba JPB",
    version="0.1",
    description="API personal testing",
    # dependencies=[Depends(verify_token)]
)

# Incluir routers
app.include_router(router)  # Asegúrate de pasar el objeto router correctamente

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
