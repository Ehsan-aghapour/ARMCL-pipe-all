
awk -F'[ :,\/]' '/NeonCL/{i=0} /CLNeon/{i=1}  /Partition layer/{Layer=$NF} /Cost/{t=$NF}
	/total_time/{t_cpu=$NF} /total2_time/{t_gpu=$NF; print i OFS Layer OFS t OFS t_cpu OFS t_gpu}' OFS=, ppp.txt > gg.csv
	
	
'''	END{
		for(ii in time){
			for(jj in time[ii]){
				out[ii][jj]=jj OFS t_cpu[ii][jj] OFS t_gpu[ii][jj] OFS time[ii][jj]
			}
		};
		for(ii in out){
			for(jj in out[ii]){
				print out[ii][jj]
			}
		};
	 }
	 '''
