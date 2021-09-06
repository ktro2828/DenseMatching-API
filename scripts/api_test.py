#!/usr/bin/env python3

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"hello": "world"}