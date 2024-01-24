<script>
	import { store } from "../../store/store";
	import { getModels } from "../api/getModels";
	import RetrainModelModal from "./RetrainModelModal.svelte";
	export let title;
	export let description;
	export let iconPath;

	const fetchModels = async () => {
		let models = await getModels();
		store.update((previousState) => ({
			...previousState,
			models,
		}));
		document.getElementById("prediction-mode-modal").showModal();
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
			<button class="btn" on:click={fetchModels}>Select</button>
			<RetrainModelModal />
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
