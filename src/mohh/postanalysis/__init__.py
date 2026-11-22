"""Scripts de analisis post-busqueda, reutilizables como paquete y por CLI propia."""

#: nombre corto -> (modulo dentro de ``mohh.postanalysis``, funcion de entrada)
COMMANDS = {
    "gen": ("analyze_gen", "main"),
    "operators": ("analyze_operators", "main"),
    "search": ("analyze_search", "main"),
    "ast-diversity": ("ast_diversity", "main"),
    "eval31": ("evaluate_31", "main"),
    "generations": ("evaluate_generations", "main"),
    "figures": ("export_figure_values", "main"),
    "reference-sets": ("generate_reference_sets_pop", "main"),
    "generation-comparison": ("generation_comparison", "main"),
    "hv-scale": ("hv_scale_from_evaluation", "main"),
    "indicators": ("indicator_quality", "main"),
    "reference-report": ("reference_sets_report", "main"),
}
