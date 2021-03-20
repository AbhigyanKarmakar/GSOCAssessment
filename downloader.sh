N=8
(
for year in $(seq 2011 2015); do
	(
	for day in $(seq -w 1 365); do 
	   ((i=i%N)); ((i++==0)) && wait
	      wget -nc -r -np -nH --cut-dirs=3 -R index.html -R index.html.tmp `echo "https://pdsimage2.wr.usgs.gov/archive/mess-e_v_h-mdis-2-edr-rawdata-v1.0/MSGRMDS_1001/DATA/""$year""_""$day/"` & 
	   sleep 2
	done
	)
done
)