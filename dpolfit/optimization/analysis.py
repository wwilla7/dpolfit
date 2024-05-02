import json
import os
from glob import glob

import jinja2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dpolfit.optimization.optimization import parameter_names
from lxml import etree
import importlib_resources

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
    "gas_mu": "Gas Phase Dipole Moment (D)",
    "condensed_mu": "Condensed Phase Dipole Moment (D)",
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
            #prp = json.load(
            #    open(os.path.join(data_path, iteration, "properties.json"), "r")
            #)
            prp = pd.read_json(os.path.join(data_path, iteration, "properties.csv"))
            params = json.load(
                open(os.path.join(data_path, iteration, "parameters.json"), "r")
            )
            #objt = prp["objective"]
            #this_data = {"iteration": i + 1, "objective": objt}
            prp["iteration"] = iteration + 1
            this_data = {k: v["value"] for k, v in params.items()}
            this_data |= prp.to_dict(orient="records")[0]
            #try:
            #    t = prp["temperature"]
            #except KeyError:
            #    tmp = json.load(
            #        open(os.path.join(data_path, "iter_001", "l", "input.json"), "r")
            #    )
            #    t = tmp["temperature"]
            #this_data |= {"temperature": float(t)}
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
    dt = dt.rename(index=tex_names)
    dt = dt.reset_index()

    styler = dt.style
    styler.hide(axis="index")
    styler.format(subset="Experiment", formatter="{:.3g}")
    styler.format(subset="Calculated", formatter="{:.3g}")
    styler.format(subset="Weight", formatter="{:.2f}")
    styler.set_properties(**{"scriptsize": "--rwrap"})

    table = styler.to_latex(hrules=True)
    table = table.replace("2.9", "2.5-3.1")

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
    data: pd.DataFrame,
    reference: pd.DataFrame,
    xplots=4,
    plot_size=4,
    re=False,
    xdata="iteration",
    ylabel="Calculated",
    label0=True,
    fontsize=16,
    othermodel=True,
    vline=None,
):
    nplots = len(reference)
    yplots = np.ceil(nplots / xplots).astype(int)
    best = data.loc[data["objective"] == data["objective"].min(), "iteration"].values[0]

    fig, axs = plt.subplots(
        yplots,
        xplots,
        figsize=(xplots * plot_size, yplots * plot_size),
        dpi=256,
        sharex=False,
    )
    xcount = 0
    ycount = 0

    fun = {
        True: lambda x, k: calc_relative_errors(x, reference.loc[k, "expt"]),
        False: lambda x, k: x,
    }

    for k in reference.index:
        if yplots > 1:
            ax = axs[ycount, xcount]
        else:
            ax = axs[xcount]

        y = fun[re](data[k], k)
        ax.scatter(data[xdata], y)

        if othermodel:
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

        else:
            if "expt" in reference.columns:
                ax.axhline(
                    y=fun[re](reference.loc[k, "expt"], k),
                    c="red",
                    ls="--",
                    label="expt",
                )

            elif vline is not None:
                ax.axvline(x=vline, ls="--", c="red", label="ref: {:.3f}".format(vline))

        if label0:
            ax.scatter(
                data.loc[data["iteration"] == best][xdata],
                fun[re](data.loc[data["iteration"] == best][k], k),
                marker="*",
                s=50,
                color="red",
                label="Best",
            )

        # add reference line

        ax.grid(ls="--")
        try:
            ax.set_title(tex_names[k], fontsize=fontsize)
        except KeyError:
            ax.set_title(k.title(), fontsize=fontsize)
        ax.set_xlabel(xdata.title(), fontsize=fontsize)
        ax.set_ylabel(ylabel.title(), fontsize=fontsize)
        ax.set_box_aspect(1.0)

        if xcount < xplots - 1:
            xcount += 1
        else:
            xcount = 0
            ycount += 1

    h, l = ax.get_legend_handles_labels()
    fig.legend(
        h,
        l,
        loc="lower left",
        fontsize=fontsize + 3,
        frameon=False,
        bbox_to_anchor=(0.9, 0.25),
    )
    [fig.delaxes(a) for a in axs.flatten()[nplots:]]
    fig.tight_layout()
    return fig


def _parameter_sub_plots(
    wd, df: pd.DataFrame = None, xplots=4, plot_size=4, fontsize=9
):

    os.chdir(wd)

    if df is None:
        df = make_dataframe("simulations")

    parameter_references = pd.read_csv(
        os.path.join("templates", "parameters.csv"), index_col="Parameter"
    )
    property_reference = pd.read_csv(
        os.path.join("templates", "references.csv"), index_col="property"
    )
    property_reference = property_reference.drop(columns=["weight"])

    for param in property_reference.index:
        fig = make_sub_plots(
            data=df,
            reference=parameter_references,
            xplots=xplots,
            plot_size=plot_size,
            re=False,
            xdata=param,
            ylabel="Parameter Value",
            fontsize=fontsize,
            othermodel=False,
            vline=property_reference.loc[param, "expt"],
        )
        fig.savefig(f"{param.replace('_', '-')}.png", dpi=256, bbox_inches="tight")


def main(wd, df: pd.DataFrame = None):
    os.chdir(wd)
    if df is None:
        df = make_dataframe(os.path.join(wd, "simulations"))

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
    property_reference = property_reference.drop(columns=["weight"])

    figs = ["prop", "rmae", "param", "objt"]

    for d, r, re, n, l, o, fs in zip(
        [df, df, df, df],
        [
            property_reference,
            property_reference,
            parameter_references,
            pd.DataFrame(index=["objective"]),
        ],
        [False, True, False, False],
        figs,
        [True, True, True, False],
        [False, True, True, False],
        [16, 16, 9, 16],
    ):
        fig = make_sub_plots(
            data=d,
            reference=r,
            xplots=4,
            plot_size=4,
            re=re,
            label0=l,
            othermodel=o,
            fontsize=fs,
        )
        fig.savefig(f"{n}.png", dpi=256, bbox_inches="tight")

    # additional analysis
    _parameter_sub_plots(wd, df=df)

    my_jinja2_env = latex_jinja_env(
        search_path=importlib_resources.files("dpolfit").joinpath(
            os.path.join("data", "templates")
        )
    )
    # my_jinja2_env = latex_jinja_env(search_path="templates")
    template = my_jinja2_env.get_template("template.tex")

    ret = template.render(
        tprop="prop.png",
        tobjt="objt.png",
        BestResult=bestresult,
        caption=f"Best result found at iteration {best}/{len(df)}",
        frametitle="Optimization",
        tabular=table,
        trmae="rmae.png",
        tparm="param.png",
        properties=[p.replace("_", "-") for p in property_reference.index],
    )

    with open("main.tex", "w") as f:
        f.write(ret)

    os.system("pdflatex main.tex")
    os.system("zathura main.pdf &")

    # df.to_csv("data.csv", index=False)


if __name__ == "__main__":
    main(os.getcwd())
