import json
import os
from glob import glob

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dpolfit.optimization.optimization import parameter_names
from lxml import etree

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    }
)

tex_names = {
    "epsilon": "Dielectric Constant",
    "rho": "Density (g/mL)",
    "hvap": "Heat of Vaporization (kJ/mol)",
    "alpha": "Thermal Expansion ($10^{-4} \\textrm{K}^{-1}$)",
    "kappa": "Isothermal Compressibility ($10^{-6} \\textrm{bar}^{-1}$)",
}

plot_colors = [
    "brown",
    "darkorange",
    "teal",
    "steelblue",
    "navy",
    "blue",
    "darkviolet",
    "purple",
]


def latex_jinja_env(search_path: str) -> jinja2.Environment:
    my_jinja2_env = jinja2.Environment(
        block_start_string="\BLOCK{",
        block_end_string="}",
        variable_start_string="\VAR{",
        variable_end_string="}",
        comment_start_string="\#{",
        comment_end_string="}",
        # loader=jinja2.BaseLoader,
        loader=jinja2.FileSystemLoader(searchpath=search_path),
    )
    return my_jinja2_env


def make_dataframe(data_path: str) -> pd.DataFrame:
    iterations = glob(f"{data_path}/iter*")
    n_iter = len(iterations)
    data = []
    for i in range(n_iter):
        iteration = f"iter_{i+1:03d}"
        try:
            prp = json.load(
                open(os.path.join(data_path, iteration, "properties.json"), "r")
            )
            params = json.load(
                open(os.path.join(data_path, iteration, "parameters.json"), "r")
            )
            objt = prp["objective"]
            this_data = {"iteration": i + 1, "objective": objt}
            this_data |= prp["properties"]
            this_data |= {k: v["value"] for k, v in params.items()}
            data.append(this_data)
        except FileNotFoundError:
            pass

    df = pd.DataFrame(data)
    return df


def calc_relative_errors(calc: np.array, ref: float) -> np.array:
    return abs(calc - ref) / ref


def property_latex(iteration_path) -> str:
    prp = json.load(open(os.path.join(iteration_path, "properties.json"), "r"))

    data = {
        "Experiment": prp["expt"],
        "Calculated": prp["properties"],
        "Weight": prp["weights"],
    }
    dt = pd.DataFrame(data)
    dt.index.name = "Property"
    dt = dt.reset_index()

    styler = dt.style
    styler.hide(axis="index")
    styler.format(subset="Experiment", formatter="{:.3g}")
    styler.format(subset="Calculated", formatter="{:.3g}")
    styler.format(subset="Weight", formatter="{:.2f}")
    styler.set_properties(**{"scriptsize": "--rwrap"})

    table = styler.to_latex(hrules=True)

    return table


def get_all_parameters_from_xml(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()

    data = {}
    for k, v in parameter_names.items():
        ret = root.find(k)
        data[k] = float(ret.attrib[v])

    return data


def parameter_latex(iteration_path: str) -> str:

    data = []

    initis = get_all_parameters_from_xml(
        os.path.join(iteration_path.split("/iter")[0], "iter_001", "forcefield.xml")
    )
    finals = get_all_parameters_from_xml(os.path.join(iteration_path, "forcefield.xml"))
    final_params = json.load(open(os.path.join(iteration_path, "parameters.json"), "r"))

    for k, v in parameter_names.items():
        this_data = {
            "Parameter": k,
            "Initial": float(initis[k]),
            "Final": float(finals[k]),
        }
        try:
            this_data |= {"Prior": final_params[k]["prior"]}
        except KeyError:
            this_data |= {"Prior": 0.0}

        data.append(this_data)

    dt = pd.DataFrame(data)
    styler = dt.style
    styler.hide(axis="index")
    styler.format(subset="Prior", formatter="{:.3g}")
    styler.format(subset="Initial", formatter="{:.6g}")
    styler.format(subset="Final", formatter="{:.6g}")
    styler.apply(
        lambda x: [
            "rowcolor: {gray!30};" if v in final_params.keys() else "" for v in x
        ],
        axis=0,
    )
    styler.set_properties(**{"scriptsize": "--rwrap"})

    table = styler.to_latex(hrules=True)
    return table


def make_sub_plots(
    data: pd.DataFrame, reference: pd.DataFrame, xplots=4, plot_size=4, re=False
):
    nplots = len(reference)
    yplots = np.ceil(nplots / xplots).astype(int)

    fig, axs = plt.subplots(
        yplots,
        xplots,
        figsize=(xplots * plot_size, yplots * plot_size),
        dpi=256,
        sharex=True,
    )
    xcount = 0
    ycount = 0

    fun = {True: lambda x, k: calc_relative_errors(x, reference.loc[k, "expt"]),
           False: lambda x, k: x}

    for k in reference.index:
        if yplots > 1:
            ax = axs[ycount, xcount]
        else:
            ax = axs[xcount]

        y = fun[re](data[k], k)

        [
            (
                ax.axhline(
                    y=fun[re](y, k),
                    c=plot_colors[idx],
                    ls="--",
                    label=l,
                )
                if l != "expt"
                else ""
            )
            for idx, (y, l) in enumerate(
                zip(reference.loc[k].values, reference.loc[k].index)
            )
        ]

        ax.scatter(data["iteration"], y)

        # add reference line

        ax.grid(ls="--")
        try:
            ax.set_title(tex_names[k])
        except KeyError:
            ax.set_title(k)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Calculated")
        ax.set_box_aspect(1.0)

        if xcount < xplots - 1:
            xcount += 1
        else:
            xcount = 0
            ycount += 1

    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", fontsize=16)
    [fig.delaxes(a) for a in axs.flatten()[nplots:]]
    fig.tight_layout()
    return fig


def main(wd):
    os.chdir(wd)
    df = make_dataframe("simulations")
    best = df.loc[df["objective"] == df["objective"].min(), "iteration"].values[0]
    optimal_path = os.path.join("simulations", f"iter_{best:03d}")
    bestresult = property_latex(optimal_path)
    table = parameter_latex(optimal_path)
    parameter_references = pd.read_csv(
        os.path.join("templates", "parameters.csv"), index_col="Parameter"
    )
    property_reference = pd.read_csv(
        os.path.join("templates", "references.csv"), index_col="property"
    )

    figs = ["prop", "rmae", "param"]

    for d, r, re, n in zip(
        [df, df, df],
        [property_reference, property_reference, parameter_references],
        [False, True, False],
        figs,
    ):
        fig = make_sub_plots(data=d, reference=r, xplots=4, plot_size=4, re=re)
        fig.savefig(f"{n}.png", dpi=256)

    my_jinja2_env = latex_jinja_env(search_path="templates")
    template = my_jinja2_env.get_template("template.tex")

    ret = template.render(
        BestResult=bestresult,
        caption=f"iteration {best}/{len(df)}",
        frametitle="Optimization",
        tabular=table,
        tprop="prop.png",
        trmae="rmae.png",
        tparm="param.png",
    )

    with open("main.tex", "w") as f:
        f.write(ret)

    os.system("pdflatex main.tex")
    os.system("open main.pdf")


if __name__ == "__main__":
    main(os.getcwd())
