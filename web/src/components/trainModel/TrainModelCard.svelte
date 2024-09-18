<script>
	import { store } from "../../store/store";
	import { getModelArchitecures } from "../api/getModelArchitectures";
	import TrainModelModal from "./TrainModelModal.svelte";
	export let title;
	export let description;
	export let iconPath;

	const fetchModelTypes = async () => {
		const modelArchitectures = await getModelArchitecures();
		store.update((previousState) => ({
			...previousState,
			modelArchitectures: modelArchitectures.data,
		}));
		document.getElementById("train-model-modal").showModal();
	};
</script>

<div class="card w-96 bg-base-100 shadow-xl">
	<figure>
		<img width="182px" src={iconPath} alt="Icon" />
	</figure>
	<div class="card-body">
		<h2 class="card-title">{title}</h2>
		<p>{description}</p>
		<div class="card-actions justify-end">
			<button class="btn" on:click={fetchModelTypes}>Select</button>
			<TrainModelModal />
		</div>
	</div>
</div>

<style>
	p {
		text-align: start;
	}

	figure {
		padding-top: 5rem;
		padding-bottom: 5rem;
	}

	.card {
		width: 50%;
	}
</style>
