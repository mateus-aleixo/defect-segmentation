# Deploying to AWS

The service runs on Lambda behind an HTTP API, from the same container that runs
locally, on the stack [turbofan-rul](https://github.com/mateus-aleixo/turbofan-rul) uses: ECR, Lambda, HTTP
API Gateway, Terraform, and GitHub Actions authenticating by OIDC with no stored
keys. The AWS resources keep the project's original name, `conformal-seg`.

Live: `https://6s8ozlsit4.execute-api.eu-west-1.amazonaws.com`

## What it costs

| Service | Free tier | This project |
|---|---|---|
| Lambda | 1M requests + 400k GB-s / month, forever | ~0 at demo traffic |
| API Gateway (HTTP) | 1M requests / month, first 12 months | ~0; $1/M after year 1 |
| ECR | 500 MB storage, first 12 months | image ≈ 0.52 GB → ~$0.05/month after year 1 |
| CloudWatch logs | 5 GB ingest / month | ~0 |

The stage is throttled to 5 req/s (burst 10) and a $5/month budget alarm emails at
80%, so the worst case is capped twice over.

Measured: **cold start ~12 s, warm ~0.45 s**. Cold start is dominated by pulling a
520 MB image and loading a MobileNetV3 graph into onnxruntime. The function runs at
2048 MB, double rul's, because Lambda scales vCPU with memory and this service does
real convolution rather than a 30x24 sensor window; more memory finishes sooner, so
it is not more money.

## Two things that differ from turbofan-rul

**The OIDC provider is shared, not created.** An IAM OIDC provider is account-global
and keyed by URL. turbofan-rul's stack already created
`token.actions.githubusercontent.com` in this account, so `infra/oidc.tf` here reads
it with a `data` block. Declaring it as a `resource` a second time fails with
`EntityAlreadyExists`, and worse, a `terraform destroy` in this repo would delete the
provider out from under rul's deploy role. Against a fresh account, apply rul's infra
first or flip that block back to a resource.

**The registry is not in git.** rul commits its `models/` because its networks are a
few hundred KB. A MobileNetV3 backbone is 42 MB per category, so this registry ships
as the `registry-v1` release asset, and `deploy.yml` fetches and extracts it before
building. The image therefore carries real networks while the repository stays small.
Rebuild the asset after retraining:

```bash
python -m conformal_seg.registry --run metal_nut
tar -czf registry.tar.gz models/metal_nut models/grid
gh release upload registry-v1 registry.tar.gz --clobber
```

## Bootstrap from zero

The Lambda references a container image, so ECR must exist and hold one image before
the first full apply:

```powershell
cd infra
terraform init
terraform apply -target=aws_ecr_repository.api -var budget_email=YOU@example.com

$acct = aws sts get-caller-identity --query Account --output text
$reg  = "$acct.dkr.ecr.eu-west-1.amazonaws.com"
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $reg

python -m conformal_seg.registry --run metal_nut   # models/ must exist before build
docker build -t "$reg/conformal-seg:latest" ..
docker push "$reg/conformal-seg:latest"

terraform apply -var budget_email=YOU@example.com
```

`terraform output` prints `api_endpoint`, `ecr_repository_url`,
`lambda_function_name` and `deploy_role_arn`.

## Wire up GitHub deploys

```powershell
gh variable set AWS_ROLE_ARN --body (terraform output -raw deploy_role_arn)
gh variable set AWS_REGION   --body eu-west-1
gh variable set API_ENDPOINT --body (terraform output -raw api_endpoint)
```

After that every deploy is `git tag v0.2.0 && git push --tags`, or
`gh workflow run deploy`.

## Verify

```bash
curl $API/health
curl $API/models
curl -X POST "$API/predict?category=metal_nut" -F image=@part.png
```

The deploy job's smoke test checks `/models` reports `metal_nut`, not just that
`/health` is up: an image built without a registry starts happily and serves 503 on
every prediction, which is a failed deploy that a health check alone calls green.

## Teardown

```powershell
terraform destroy -var budget_email=YOU@example.com
```

The ECR repository is `force_delete`, so images go with it. Note the shared OIDC
provider is a `data` source here and is therefore left alone, which is the intended
behaviour: it belongs to turbofan-rul's stack.
