export async function triggerTraining(
	architecture,
	name,
	description,
	smilesCsvFile, 
	parameters
) {
	const data = new FormData();

	data.append("name", name)
	data.append("modelArchitecture", JSON.stringify(architecture));
	data.append("dataFile", smilesCsvFile);

	if (Object.keys(parameters)) {
		data.append("parameters", JSON.stringify(parameters));
	}
	
	if (description) {
		data.append("description", description);
	}

	const res = await fetch("http://localhost:3000/trigger-training", {
		method: "POST",
		body: data,
	});

	const json = await res.json();
	return json;
}
