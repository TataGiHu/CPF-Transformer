## docker 启动

```

image_name=artifactory.momenta.works/docker-momenta/dps/dps_dev_wjw:v1.0.2
container_name=xxx

docker run -dit \
	--name=$container_name \
	--runtime=nvidia  \
	--shm-size=1024m \
	$image_name  /bin/bash

docker exec -it $container_name bash

```


## Run training demo

- Data Parallel (DP) training 

```

	cd tools
	bash run_dp.sh

```


- Data Distributed Parallel (DDP) training 

```

	cd tools 
	bash run_ddp.sh

```

