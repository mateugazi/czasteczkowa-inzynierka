export const scrollToTop = () =>
	document
		.getElementsByClassName("main-app")[0]
		.scrollIntoView({ behavior: "smooth" });
