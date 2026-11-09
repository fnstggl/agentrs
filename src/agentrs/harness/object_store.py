"""Artifact storage (screenshots, traces, encrypted browser session state): S3-compatible
(MinIO / AWS S3 via boto3) or a local directory."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


class ObjectStore:
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str: ...
    def get(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class LocalObjectStore(ObjectStore):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("key escapes the store")
        return p

    def put(self, key, data, content_type="application/octet-stream"):
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return f"file://{key}"

    def get(self, key):
        return self._p(key).read_bytes()

    def delete(self, key):
        self._p(key).unlink(missing_ok=True)

    def exists(self, key):
        return self._p(key).exists()


class S3ObjectStore(ObjectStore):
    def __init__(self, bucket: str, endpoint_url: Optional[str] = None, region: str = "us-east-1", client=None):
        import boto3

        self.bucket = bucket
        self.s3 = client or boto3.client("s3", endpoint_url=endpoint_url, region_name=region)
        try:
            self.s3.head_bucket(Bucket=bucket)
        except Exception:  # noqa: BLE001 - create in development (MinIO / moto); production buckets are pre-created
            self.s3.create_bucket(Bucket=bucket)

    def put(self, key, data, content_type="application/octet-stream"):
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type,
                           Metadata={"sha256": hashlib.sha256(data).hexdigest()})
        return f"s3://{self.bucket}/{key}"

    def get(self, key):
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def exists(self, key):
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # noqa: BLE001
            return False
