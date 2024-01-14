export async function getModelTypes() {
	const res = await fetch("http://localhost:3000/model-type", {
		method: "GET",
	});

	const json = await res.json();
	return json;
}
