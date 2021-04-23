import numpy as np
import PIL
import io
from corner import corner
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots
from itertools import combinations
from skimage.transform import downscale_local_mean

#%%
def images_expo(cluster):
    cbarlocs = [.285, .64, .995]

    img = cluster.dat.img.copy()
    exp = cluster.dat.exposure.copy()
    bkg = cluster.dat.bkg.copy()

    index = exp==0

    img[index] = np.nan
    exp[index] = np.nan
    bkg[index] = np.nan

    img = downscale_local_mean(img, (3,3))
    img[img == 0] = np.nan
    exp = downscale_local_mean(exp, (3,3))
    bkg = downscale_local_mean(bkg, (3,3))

    ra, dec = cluster.wcs.array_index_to_world_values(*np.indices(cluster.model_best_fit_image.shape))
    ra = downscale_local_mean(ra, (3, 3))
    dec = downscale_local_mean(dec, (3, 3))

    ratio = cluster.prof.ellratio
    angle = cluster.prof.ellangle*np.pi/180

    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))

    t = np.linspace(0,2*np.pi,100)
    x_ell = cluster.t500/60*np.cos(t)
    y_ell = ratio*cluster.t500/60*np.sin(t)
    
    x_ell, y_ell = R@np.vstack((x_ell, y_ell))

    fig = make_subplots(rows=1, cols=3,subplot_titles=('Data', 'Exposure', 'Background'),shared_xaxes='all',shared_yaxes='all')

    fig.add_trace(
        go.Heatmap(x = ra[0,:],
                   y = dec[:,0],
                   z=np.log10(img),
                   hoverongaps=False,
                   name="img",
                   colorbar_x=cbarlocs[0],
                   colorbar_title='log(Count)',
                   text=['Count Map'],
                   colorscale='Jet'), row=1, col=1)

    fig.add_trace(
        go.Heatmap(x = ra[0,:],
                   y = dec[:,0],
                   z=exp/1000,
                   hoverongaps=False,
                   name="exp",
                   colorbar_x=cbarlocs[1],
                   colorbar_title='ks',
                   text=['Exposure Map'],
                   colorscale='Jet'), row=1, col=2)

    fig.add_trace(
        go.Heatmap(x = ra[0,:],
                   y = dec[:,0],
                   z=bkg, hoverongaps=False,
                   name="bkg",
                   colorbar_x=cbarlocs[2],
                   colorbar_title='Count',
                   text=['Background Map'],
                   colorscale='Jet'), row=1, col=3)

    fig.add_trace(go.Scatter(
        x=[cluster.prof.cra],
        y=[cluster.prof.cdec],
        name='c',
        text=['Estimated center'],
        marker_symbol="x",
        marker_color="purple"
    ), row=1, col="all")

    fig.add_trace(go.Scatter(
        x=x_ell + cluster.prof.cra,
        y=y_ell + cluster.prof.cdec,
        name='r500',
        text=['R500 Ellipse']
    ), row=1, col="all")

    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(showlegend=False)
    fig.update_layout(autosize = True)
    fig.update_layout(title='Raw inputs')
    #fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False, xaxis_zeroline=False, yaxis_zeroline=False)
    #fig.write_html("images.html", include_mathjax='cdn')
    #plot(fig, include_mathjax='cdn')

    return plot(fig, include_mathjax='cdn', output_type='div')

#%% Profile de surface de brillance
def profile(cluster):

    rads = cluster.prof.bins
    prof = cluster.prof.profile

    tmod = cluster.mod(rads, *cluster.mod.params)
    chi = (prof-cluster.mod(rads, *cluster.mod.params))/cluster.prof.eprof

    fig = make_subplots(rows=2, cols=2,shared_xaxes=True,
                        row_width=[0.2, 0.8],
                        specs=[[{}, {"rowspan": 2}],
                               [{}, None]],
                        vertical_spacing=0.02
                        )


    fig.add_scatter(x=rads, y=prof, mode='markers', name='SB',
                            error_x=dict(type='data',
                                          array=cluster.prof.ebins,
                                          visible=True),
                            error_y=dict(type='data',
                                         array=cluster.prof.eprof,
                                         visible=True),
                            row=1, col=1)

    fig.add_scatter(x=rads, y=tmod, name='BetaModel', row=1, col=1,
                    error_x=dict(type='data',
                                 array=cluster.prof.ebins,
                                 visible=True,
                                 thickness=0.),
                    )

    fig.add_scatter(x=rads, y=cluster.prof.bkgprof, name='Background', row=1, col=1,
                    error_x=dict(type='data',
                                 array=cluster.prof.ebins,
                                 visible=True,
                                 thickness=0.),
                    )

    fig.add_trace(go.Scatter(x=rads, y=chi, mode='markers', name=r'$\chi$',
                            error_y=dict(type='data',
                                         array=np.ones(len(rads)),
                                         visible=True)),
                            row=2, col=1)

    ra, dec = cluster.wcs.array_index_to_world_values(*np.indices(cluster.model_best_fit_image.shape))
    ra = downscale_local_mean(ra, (3, 3))
    dec = downscale_local_mean(dec, (3, 3))
    bestmod = downscale_local_mean(cluster.model_best_fit_image, (3, 3))

    fig.add_trace(
        go.Heatmap(x=ra[0, :],
                   y=dec[:, 0],
                   z=np.log10(bestmod),
                   hoverongaps=False,
                   name="img",
                   colorbar_title='log(Count)',
                   text=['Count Map'],
                   colorscale='Jet'), row=1, col=2)

    fig.update_xaxes(type="log", row=1,col=1)
    fig.update_yaxes(type="log", title_text=r'$\text{Surface Brightness } [\text{cts }.\text{s}^{-1}.\text{arcmin}^2]$', automargin=True,row=1,col=1)
    fig.update_xaxes(type="log", title_text='Radius [arcmin]', automargin=True,row=2,col=1)
    fig.update_yaxes(title_text=r'$\chi$', automargin=True,row=2,col=1)
    fig.update_xaxes(range=np.log10([0.9*rads[0], 1.1*rads[-1]]))
    fig.update_layout(hovermode='x unified')
    fig.update_layout(
        title='Surface brightness profile Fitting',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'power'
        )
    )

    #fig.update_layout(legend=dict(
    #    yanchor="top",
    #    y=0.99,
    #    xanchor="rig",
    #    x=0.5
    #))

    fig.update_layout(
        height=800
    )

    #fig.write_html("profile.html", include_mathjax='cdn')
    #fig.add_trace(go.Table(header=dict(values=list(cluster.mod.parnames)),
    #                        cells=dict(values=list(np.round(cluster.mod.params,decimals=2)))),
    #                      row=2,col=2)

    return plot(fig, include_mathjax='cdn', output_type='div')

#%%
def power_spectrum(cluster):

    k = cluster.psc.k
    ps2d = cluster.psc.ps
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=k, y=ps2d - np.diag(cluster.ps_covariance) ** 0.5,
                             fill=None,
                             line_color='#636efa',
                             mode='lines'
                             ))

    fig.add_trace(go.Scatter(x=k, y=ps2d+np.diag(cluster.ps_covariance)**0.5,
                             fill='tonexty',
                             line_color='#636efa',
                             mode='lines'  # override default markers+lines
                             ))

    fig.add_scatter(x=k, y=cluster.psc.psnoise, name='PSNoise',
                    mode='lines',
                    line_color='red')

    fig.add_scatter(x=k, y=np.diag(cluster.ps_cov_profile)**0.5, name='Err-Profile',
                    mode='lines',
                    line_color='red')

    fig.add_scatter(x=k, y=np.diag(cluster.ps_cov_poisson)**0.5, name='Err-Poisson',
                    mode='lines',
                    line_color='green')

    fig.add_scatter(x=k, y=np.diag(cluster.ps_cov_sample) ** 0.5, name='Err-Sample',
                    mode='lines',
                    line_color='purple')

    fig.add_scatter(x=k, y=ps2d, name='$P_{2D}$',
                    mode='lines',
                    line_color='blue')

    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log", title_text=r'$\text{Power Spectrum (2D) } [\text{cts }.\text{s}^{-1}.\text{arcmin}^2]$', automargin=True)
    fig.update_xaxes(type="log", title_text='$k [\text{kpc}^{-1}]$', automargin=True)
    #fig.update_xaxes(range=np.log10([0.9*rads[0], 1.1*rads[-1]]))
    fig.update_layout(hovermode='x unified')
    fig.update_layout(
        title='Power spectrum extraction',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'power'
        )
    )

    #fig.update_layout(legend=dict(
    #    yanchor="top",
    #    y=0.99,
    #    xanchor="rig",
    #    x=0.5
    #))

    fig.update_layout(
        height=800
    )

    #fig.write_html("profile.html", include_mathjax='cdn')
    #fig.add_trace(go.Table(header=dict(values=list(cluster.mod.parnames)),
    #                        cells=dict(values=list(np.round(cluster.mod.params,decimals=2)))),
    #                      row=2,col=2)

    return plot(fig, include_mathjax='cdn', output_type='div')

#%% Corner plot
def corner_plotly(cluster):

    fig = make_subplots(rows=cluster.mod.npar-1, cols=cluster.mod.npar-1, shared_xaxes=True)
    df = pd.DataFrame(data=cluster.model_samples, columns=cluster.mod.parnames)
    for var1, var2 in combinations(cluster.mod.parnames, 2):

        fig.add_trace(go.Histogram2dContour(
            x=df[var1],
            y=df[var2],
            colorscale='Blues',
            histnorm='probability'
        ), row=cluster.mod.parnames.index(var2), col=cluster.mod.parnames.index(var1)+1)

    fig.update_traces(showscale=False)
    fig.update_layout(autosize = True)
    fig.update_layout(
        width=1400,
        height=1000
    )
    fig.write_html("corner.html", include_mathjax='cdn')


    #return fig

#%%
def header(cluster):
    fig = go.Figure(data=[go.Table(header=dict(values=['Name', 'R500 (arcmin)', 'R500 (kpc)', 'Redshift', 'Tag']),
                                   cells=dict(values=[cluster.name, cluster.r500_arcmin, cluster.r500, cluster.z, cluster.tag]))
                          ])
    fig.update_layout(
        width=1400,
        height=300
    )
    return plot(fig, include_mathjax='cdn', output_type='div')

#%%

def dashboard(cluster, outfile='multi_plot.html'):
    
    hdr = header(cluster)
    div2 = images_expo(cluster)
    div3 = profile(cluster)
    div4 = power_spectrum(cluster)

    html = """\
    <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
        {}
        {}
        {}
        {}
        </body>
    </html>
    """.format(hdr, div2, div3, div4)

    with open(outfile, 'w') as f:
        f.write(html)