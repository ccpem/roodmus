"""Heterogeneity reconstruction example class. May be deprecated.

Delft University of Technology (TU Delft) hereby disclaims
all copyright interest in the program “Roodmus” written by
the Author(s).
Copyright (C) 2023  Joel Greer(UKRI), Tom Burnley (UKRI),
Maarten Joosten (TU Delft), Arjen Jakobi (TU Delft)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from roodmus.analysis.utils import IO


class HetRec(object):
    def __init__(self):
        pass

    @classmethod
    def add_latent_space_coordinates(
        self,
        latent_file: str,
        df_picked: pd.DataFrame,
    ):
        """Add latent space coordinates obtained through
        heterogeneous reconstruction to the
        picked particle data frame.
        """

        # check by the file extension which method
        # was used to obtain the latent space coordinates
        if latent_file.endswith(".cs"):
            # cryoSPARC
            latent_space, ndim = IO.get_latents_cs(latent_file)

        elif latent_file.endswith(".pkl"):
            # cryoDRGN
            latent_space, ndim = IO.get_latents_cryodrgn(latent_file)

        else:
            raise ValueError(f"unknown latent space file type: {latent_file}")

        # add the latent space coordinates to the picked particle data frame
        for i in range(ndim):
            df_picked["latent_{}".format(i)] = latent_space[:, i]
        return df_picked, ndim

    @classmethod
    def compute_PCA(
        self,
        df_picked: pd.DataFrame,
        ndim: int,
    ):
        pca = PCA(n_components=ndim)
        pca.fit(df_picked[["latent_{}".format(i) for i in range(ndim)]])
        x = pca.fit(df_picked[["latent_{}".format(i) for i in range(ndim)]])
        results = x.transform(
            df_picked[["latent_{}".format(i) for i in range(ndim)]]
        )

        # add the PCA coordinates to the picked particle data frame
        for i in range(ndim):
            df_picked["PCA_{}".format(i)] = results[:, i]

        return df_picked, pca

    @classmethod
    def compute_TSNE(
        self,
        df_picked: pd.DataFrame,
        ndim: int,
    ):
        tsne = TSNE(n_components=ndim)
        results = tsne.fit_transform(
            df_picked[["latent_{}".format(i) for i in range(ndim)]]
        )

        # add the t-SNE coordinates to the picked particle data frame
        for i in range(ndim):
            df_picked["tSNE_{}".format(i)] = results[:, i]

        return df_picked, tsne

    @classmethod
    def compute_UMAP(
        self,
        df_picked: pd.DataFrame,
        ndim: int,
    ):
        pass

    @classmethod
    def compute_kmeans(
        self,
        df_picked: pd.DataFrame,
        ndim: int,
        n_clusters: int = 3,
        pca: bool = False,
        tsne: bool = False,
        umap: bool = False,
    ):
        if pca:
            varname = "PCA"
        elif tsne:
            varname = "tSNE"
        elif umap:
            varname = "UMAP"
        else:
            varname = "latent"

        kmeans = KMeans(n_clusters=n_clusters).fit(
            df_picked[["{}_{}".format(varname, i) for i in range(ndim)]]
        )
        df_picked["kmeans"] = kmeans.labels_
        return df_picked, kmeans
