export async function getPredictions(_id, smilesCsvFile) {
	const data = new FormData();
	data.append("_id", _id);
	data.append("file", smilesCsvFile);
	const res = await fetch("http://localhost:3000/get-predictions", {
		method: "POST",
		body: data,
	});

	const json = await res.json();
	return json.data;
}
