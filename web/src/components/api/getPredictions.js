export async function getPredictions(smilesCsvFile) {
	const data = new FormData();
	data.append("uniqueName", "model_GBT_pipeline");
	data.append("file", smilesCsvFile);
	const res = await fetch("http://localhost:3000/get-predictions", {
		method: "POST",
		body: data,
	});

	const json = await res.json();
	return json;
}