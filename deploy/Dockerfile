## 指定java 运行环境镜像

FROM busybox:latest

LABEL maintainer="MY_diagnosis_guazhou"
LABEL desc="云集控能效评估及诊断"

ARG install_dir=/app
ARG project_name=MY_diagnosis
RUN pip3 install tornado
#ADD ${project_name}/target/${project_name}-bundle-tar.tar.gz ${install_dir}/
#这里需要把项目里的lib包下SNAPSHOT包放到最后写入docker image, 因为SNAPSHOT是不稳定的
#其他的jar 是稳定的，这样做能最大化利用镜像的缓存功能，减少镜像的空间
COPY tmp/* ${install_dir}/${project_name}/


WORKDIR ${install_dir}/${project_name}/


