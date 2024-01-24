<script>
	import { store } from "../../store/store";
	import { getModels } from "../api/getModels";
	export let title;
	export let description;
	export let iconPath;
	export let viewMode;

	let models;

	const unsubscribe = store.subscribe((state) => {
		models = state.models;
	});

	const fetchModels = async () => {
		models = await getModels();
		store.update((previousState) => ({
			...previousState,
			models,
		}));
		document.getElementById("my_modal_4").showModal();
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
			<dialog id="my_modal_4" class="modal">
				<div class="modal-box w-11/12 max-w-5xl">
					<h2 class="font-bold text-3xl">Select model</h2>
					{#if models && models.length > 0}
						{#each models as model}
							<div class="card w-96 bg-base-100 shadow-xl">
								<div class="card-body">
									<h2 class="card-title">{model.name}</h2>
									<p>{model.description}</p>
									<div class="card-actions justify-end">
										<button
											class="btn btn-primary"
											on:click={() =>
												store.update((previousState) => ({
													...previousState,
													viewMode,
												}))}>Select</button
										>
									</div>
								</div>
							</div>
						{/each}
					{:else}
						<p class="py-4">Sorry, no models available</p>
					{/if}
					<div class="modal-action">
						<form method="dialog">
							<!-- if there is a button, it will close the modal -->
							<button class="btn">Close</button>
						</form>
					</div>
				</div>
			</dialog>
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
