export async function getModelArchitecures() {
	const res = await fetch("http://localhost:3000/model-architecture", {
		method: "GET",
	});

	const json = await res.json();
	return json;
}
