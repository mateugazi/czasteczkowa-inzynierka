export async function triggerTraining(smilesCsvFile, parameters, modelType) {
	const data = new FormData();
	data.append("parameters", JSON.stringify(parameters));
	data.append("modelType", modelType);
	data.append("file", smilesCsvFile);
	const res = await fetch("http://localhost:3000/trigger-training", {
		method: "POST",
		body: data,
	});

	const json = await res.json();
	return json;
}
