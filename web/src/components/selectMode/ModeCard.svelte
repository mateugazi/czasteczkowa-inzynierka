<script>
  export let title;
  export let description;
  export let iconPath;
</script>

<div class="card w-96 bg-base-100 shadow-xl">
  <figure>
    <img width="182px" src={iconPath} alt="Icon" />
  </figure>
  <div class="card-body">
    <h2 class="card-title">{title}</h2>
    <p>{description}</p>
    <div class="card-actions justify-end">
      <button class="btn btn-primary">Select</button>
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