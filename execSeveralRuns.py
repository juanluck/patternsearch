from subprocess import run, Popen, PIPE
from pathlib import Path
import multiprocessing
import time
from random import random
from datetime import datetime
from datetime import timedelta


# global variables
octaveScriptName = "patternsearch.sh"
x0v = [2,3]
alpha0v = [0.1]
objectivefunctionv = ["sphere","ellipsoid"]
basisv = ["standard"]
orderv = ["same"]
tauplusv = [1]
tauminusv = [0.5]


def nbContainers():
	result=run("docker ps -aq | wc -l", shell=True, stdout=PIPE).stdout.decode('utf-8')
	return int(result)
    
def prepareExperiments():
	listExper = []
	for x0 in x0v:
		strx0 = "\'"
		for x in range(x0):
			strx0 += str(10 * random() - 5)+" "
		strx0 += "\'"
		for alpha0 in alpha0v:
			for objectivefunction in objectivefunctionv:
				for basis in basisv:
					for order in orderv:
						for tauplus in tauplusv:
							for tauminus in tauminusv:
								currentExper = [strx0,str(alpha0),objectivefunction,basis,order,str(tauplus),str(tauminus)]
								listExper.append(currentExper)
	return listExper
        


# Starting date
startingDate = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# The number of available CPU
nbCpus = multiprocessing.cpu_count()

# The date of the last backup done
lastBackup = datetime.now()

NoSIM = 2
run("mkdir Results", shell = True)

for nbSimStarted in range(NoSIM):
	
	listExper = prepareExperiments()
	print(str(nbSimStarted+1)+"/"+str(NoSIM)+" Experimets in this simulation: "+str(len(listExper)))
	for exper in listExper:
		
		outputFile = "Results/results_"+ str(exper[0].count(' '))+ "_"+ exper[1]+ "_"+ exper[2]+ "_"+ exper[3]+ "_"+ exper[4]+ "_"+ exper[5]+ "_"+ exper[6]
		if not Path(outputFile).is_file():
			run("echo \"Time,iterations,evaluations,fitness\" > "+outputFile, shell = True)
		print(exper)
		command = ["docker run --rm --entrypoint octave octave patternsearch.sh "+ exper[0]+ " "+ exper[1]+ " "+ exper[2]+ " "+ exper[3]+ " "+ exper[4]+ " "+ exper[5]+ " "+ exper[6]+" 2>/dev/null"]
		command[0] += " >> " + outputFile + " 2>/dev/null"

		Popen(command, shell = True)
		time.sleep(5) # wait 30 sec to make sure the container is really started

		# Loop if the current number of running containers is greater or equal than the number of available cpus
		while nbCpus <= nbContainers():
			# Sleep a bit: don't need to be awaken all the time
			time.sleep(350)

while nbContainers() > 0:
	# We started all the containers, but we still have to wait until their end
	time.sleep(30)
