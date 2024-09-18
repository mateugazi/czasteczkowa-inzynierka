export const retrainModel = async (_id, smilesCsvFile) => {
  const data = new FormData();
	data.append("_id", _id);
	data.append("file", smilesCsvFile);
	const res = await fetch("http://localhost:3000/retrain-model", {
		method: "POST",
		body: data,
	});

	const json = await res.json();
	return json;
}
