## Kubeflow KFServing

Useful commands, used in the Kfserving workshop:

- Get the Knative ingress port:
`kubectl get svc istio-ingressgateway --namespace istio-system --output 'jsonpath={.spec.ports[?(@.port==80)].nodePort}'`

- Send prediction request:
`curl -v -H "Host: iris.kubeflow-user-example-com.zico.biz" http://localhost:31489/v1/models/iris:predict -d @./iris-input.json`


- Send prediction request with authnticaion:
`curl -v -H "Host: iris.kubeflow-user-example-com.zico.biz" -H "Cookie: authservice_session=${SESSION}" http://localhost:31489/v1/models/iris:predict -d @./iris-input.json`
