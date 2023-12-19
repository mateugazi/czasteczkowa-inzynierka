<script>
	export let title;
	export let description;
	export let iconPath;
	export let onSelect = function () {
		console.log("// TO BE DONE");
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
			<button class="btn btn-primary" on:click={onSelect}>Select</button>
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
</style>
