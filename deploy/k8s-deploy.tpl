#这是部署k8s的模版文件，使用时需要替换{}变量，生成yaml 文件，使用kubectl apply 进行部署更新
apiVersion: apps/v1
kind: Deployment
metadata:
  namespace: {namespace}
  name: {app_name}    # deployment的名称，全局唯一
  labels:
    app: {app_name}
spec:
  replicas: {replicas}   # Pod副本的期待数量
  selector:
    matchLabels:
      app: {app_name}  # 符合目标的Pod拥有此标签
  template:       # 根据此模板创建Pod的副本
    metadata:
      labels:     # Pod副本拥有的标签
        app: {app_name}
    spec:
      tolerations:
        - key: "project.name"
          operator: "Equal"
          value: "cc"
          effect: "PreferNoSchedule"
      volumes:
        - name: file-data
          persistentVolumeClaim:
            claimName: cc-root-pvc
      restartPolicy: Always
      containers:           # Pod内容器的定义部分
        - name: {app_name}        # 容器对应的名称
          image: {image}      # 容器对应的Docker镜像
          imagePullPolicy: IfNotPresent
          resources:
            requests:
              cpu: 100m
              memory: 100Mi
            limits:
              cpu: 1000m
              memory: 1000Mi
          ports:
            - containerPort: 10419     # 容器应用监听的端口号
          env:
            - name: ENV
              value: {ENV}
            - name: TZ
              value: Asia/Shanghai
          volumeMounts:
            - name: file-data
              mountPath: /usr/local/consumer/file/csv
          lifecycle:
            preStop:
              exec:
                command: ["sh", "-c", "sleep 10"]  # set prestop hook
      terminationGracePeriodSeconds: 45    # terminationGracePeriodSeconds


---
#只有需暴露外部访问的服务才需要配置service
kind: Service
apiVersion: v1
metadata:
  labels:
    app: {app_name}-service
  name: {app_name}-service
  namespace: {namespace}
spec:
  ports:
    - port: 10419
      targetPort: 10419
  selector:
    app: {app_name}
  type: ClusterIP