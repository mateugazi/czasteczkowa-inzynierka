import { writable } from "svelte/store";

export const store = writable({
	viewMode: "selectMode",
	predictions: [],
	models: [],
	modelArchitectures: [],
	trainingResults: []
});
