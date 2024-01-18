<script>
	import { store } from "../store/store";

	function handleLogoClick() {
		store.update((state) => ({ ...state, viewMode: "selectMode" }));
	}
</script>

<div class="navbar-container">
	<div class="navbar bg-primary text-primary-content">
		<button class="btn btn-ghost text-xl" on:click={handleLogoClick}
			>CheML</button
		>
	</div>
</div>

<style>
	.navbar-container {
		display: flex;
		/* position: sticky; */
		/* top: 0; */
	}

	.navbar {
		border-radius: 10px;
		margin: 10px;
	}
</style>
