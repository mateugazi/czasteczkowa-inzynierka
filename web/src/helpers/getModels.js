export async function getModels() {
	const res = await fetch("http://localhost:3000/model", {
		method: "GET",
	});

	const json = await res.json();
	return json;
}
