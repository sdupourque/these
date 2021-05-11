import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from itertools import cycle
import os
import plotly.express as px

#%%
def images_expo(cluster):

    cbarlocs = [.285, .64, .995]

    img = cluster.dat.img.copy()
    exp = cluster.dat.exposure.copy()
    bkg = cluster.dat.bkg.copy()
    ra, dec = cluster.wcs.all_pix2world(*np.indices(cluster.dat.img.shape), 0)

    index = exp==0

    img[index] = np.nan
    exp[index] = np.nan
    bkg[index] = np.nan

    ra = ra[:,ra.shape[1]//2]
    dec = dec[dec.shape[0]//2,:]

    ratio = cluster.prof.ellratio
    angle = cluster.prof.ellangle*np.pi/180

    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))

    t = np.linspace(0,2*np.pi,100)
    x_ell = cluster.t500/60*np.cos(t)
    y_ell = ratio*cluster.t500/60*np.sin(t)
    
    x_ell, y_ell = R@np.vstack((x_ell, y_ell))

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('Data', 'Exposure', 'Background'),
                        shared_xaxes='all',
                        shared_yaxes='all',
                        print_grid=False)

    fig.add_trace(
        go.Heatmap(x = ra,
                   y = dec,
                   z = img,
                   hoverongaps=False,
                   name="img",
                   colorbar_x=cbarlocs[0],
                   colorbar_title='Count',
                   text=['Count Map'],
                   colorscale='Jet'), row=1, col=1)

    fig.add_trace(
        go.Heatmap(x = ra,
                   y = dec,
                   z = exp/1000,
                   hoverongaps=False,
                   name="exp",
                   colorbar_x=cbarlocs[1],
                   colorbar_title='ks',
                   text=['Exposure Map'],
                   colorscale='Jet'), row=1, col=2)

    fig.add_trace(
        go.Heatmap(x = ra,
                   y = dec,
                   z= bkg,
                   hoverongaps=False,
                   name="bkg",
                   colorbar_x=cbarlocs[2],
                   colorbar_title='Count',
                   text=['Background Map'],
                   colorscale='Jet'), row=1, col=3)

    fig.add_trace(go.Scatter(
        x=[cluster.prof.cra],
        y=[cluster.prof.cdec],
        name='c',
        marker=dict(size=12,
                    line=dict(width=2,
                              color='DarkSlateGrey')),
        text=['Estimated center'],
        marker_symbol="x",
        marker_color="white"
    ), row=1, col="all")

    fig.update_layout(showlegend=False)
    fig.update_layout(autosize = True)
    fig.update_layout(title='Raw inputs')
    fig.update_xaxes(zeroline=False,
                     showticklabels=False)
    fig.update_yaxes(zeroline=False,
                     showticklabels=False)

    return plot(fig, include_mathjax='cdn', output_type='div')

#%% Profile de surface de brillance
def profile(cluster):

    rads = cluster.prof.bins
    prof = cluster.prof.profile

    fig = make_subplots(rows=2, cols=2,shared_xaxes=True,
                        row_width=[0.2, 0.8],
                        column_width=[0.55, 0.45],
                        vertical_spacing=0.02,
                        horizontal_spacing=0.02,
                        specs=[[{}, {"rowspan": 2}],
                               [{}, None]],
                        subplot_titles=('1D Radial profile', '2D Synthetic model'),
                        )

    fig.add_scatter(x=rads, y=prof, mode='markers', name='SB',
                            error_x=dict(type='data',
                                          array=cluster.prof.ebins,
                                          visible=True),
                            error_y=dict(type='data',
                                         array=cluster.prof.eprof,
                                         visible=True),
                            row=1, col=1)

    fig.add_scatter(x=rads, y=cluster.prof.bkgprof, name='Background', row=1, col=1,
                    error_x=dict(type='data',
                                 array=cluster.prof.ebins,
                                 visible=True,
                                 thickness=0.),
                    )

    if cluster.mod is not None:
        tmod = cluster.mod(rads, *cluster.mod.params)
        chi = (prof - cluster.mod(rads, *cluster.mod.params)) / cluster.prof.eprof
        fig.add_scatter(x=rads, y=tmod, name='BetaModel', row=1, col=1,
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

    fig.update_xaxes(type="log", row=1,col=1)
    fig.update_yaxes(type="log", title_text=r'$\text{Surface Brightness } [\text{cts }.\text{s}^{-1}.\text{arcmin}^2]$', automargin=True,row=1,col=1)
    fig.update_xaxes(type="log", title_text='Radius [arcmin]', automargin=True,row=2,col=1)
    fig.update_yaxes(title_text=r'$\chi$', automargin=True,row=2,col=1)
    fig.update_xaxes(range=np.log10([0.9*rads[0], 1.1*rads[-1]]),row='all',col=1)
    #fig.update_layout(hovermode='x unified')
    fig.update_layout(
        title='Surface brightness profile Fitting',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'power'
        )
    )

    bestmod = cluster.model_best_fit_image.copy()
    exp = cluster.dat.exposure.copy()
    ra, dec = cluster.wcs.all_pix2world(*np.indices(cluster.dat.img.shape), 0)

    ra = ra[:, ra.shape[1] // 2]
    dec = dec[dec.shape[0] // 2, :]

    bestmod[exp == 0] = np.nan

    fig.add_trace(
        go.Heatmap(x=ra,
                   y=dec,
                   z=bestmod,
                   hoverongaps=False,
                   name="img",
                   colorbar_title='Count',
                   text=['Count Map'],
                   colorscale='Jet',
                   zmin=np.nanmin(cluster.dat.img),
                   zmax=np.nanmax(cluster.dat.img),
                   colorbar_x = .995,
                   ), row=1, col=2)

    x0, y0 = cluster.prof.cra, cluster.prof.cdec
    ratio, angle = cluster.prof.ellratio, np.radians(cluster.prof.ellangle)
    theta = cluster.t500/60

    t = np.linspace(0, 2*np.pi, 101)
    x = np.cos(t)*theta*ratio
    y = np.sin(t)*theta
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))

    x_f, y_f = np.dot(R, np.vstack((x, y)))
    x_f += x0
    y_f += y0

    fig.add_trace(go.Scatter(x=x_f,
                             y=y_f,
                             mode='lines',
                             name='r500',
                             line_color='#17becf'),
                  row=1, col=2)

    fig.update_xaxes(zeroline=False,
                     showticklabels=False,
                     row=1, col=2)

    fig.update_yaxes(zeroline=False,
                     showticklabels=False,
                     row=1, col=2)

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.55
    ))

    fig.update_layout(
        height=800,
        autosize=True
    )

    return plot(fig, include_mathjax='cdn', output_type='div')

#%%
def power_spectrum(cluster):

    k = cluster.k
    ps2d = cluster.ps

    fig = make_subplots(rows=2, cols=2,shared_xaxes=True,
                        row_width=[0.2, 0.8],
                        column_width=[0.55, 0.45],
                        vertical_spacing=0.02,
                        horizontal_spacing=0.02,
                        specs=[[{}, {"rowspan": 2}],
                               [{}, None]],
                        subplot_titles=('2D Power Spectrum', 'Fluctuation map'),
                        )

    fig.add_trace(go.Scatter(x=k, y=ps2d - np.diag(cluster.ps_covariance)**0.5,
                             fill=None,
                             line_color='#636efa',
                             mode='lines'
                             ))

    fig.add_trace(go.Scatter(x=k, y=ps2d + np.diag(cluster.ps_covariance)**0.5,
                             fill='tonexty',
                             line_color='#636efa',
                             mode='lines'  # override default markers+lines
                             ))

    fig.add_scatter(x=k, y=ps2d,
                    name='P_{2D}',
                    mode='lines',
                    line_color='blue')

    fig.add_scatter(x=k, y=np.diag(cluster.ps_cov_poisprof)**0.5,
                    name='Err-Profile/Poisson',
                    mode='lines',
                    line_dash='dot',
                    line_color='red')

    #fig.add_scatter(x=np.geomspace(min(k), max(k), 100),
    #                y=cluster.model_P3D(np.geomspace(min(k), max(k), 100), *cluster.res),
    #                name='Best fit',
    #                mode='lines',
    #                line_color='cyan', row=1, col=1)

    #chi = (ps2d - cluster.model_P3D(k, *cluster.res)) / (2*np.diag(cluster.ps_covariance)**0.5)

    #fig.add_scatter(x=k, y=chi, name=r'$\chi$', mode='markers',
    #                marker_color='#AB63FA',
    #                error_y=dict(type='data',
    #                             array=np.ones(len(cluster.psc.k)),
    #                             visible=True),
    #                row=2, col=1)

    fig.update_xaxes(type="log", col=1, row=1)
    fig.update_yaxes(type="log", title_text=r'$\text{Power Spectrum (2D) } [\text{cts }.\text{s}^{-1}.\text{arcmin}^2]$',
                     automargin=True, row=1, col=1)
    fig.update_xaxes(type="log", title_text=r'$k \text{ } [\text{kpc}^{-1}]$', automargin=True, row=2, col=1)
    fig.update_yaxes(title_text=r'$\chi$', automargin=True, row=2, col=1)
    #fig.update_xaxes(range=np.log10([0.9 * rads[0], 1.1 * rads[-1]]), row='all', col=1)

    fig.update_layout(
        title='Power spectrum extraction',
        yaxis = dict(
            showexponent = 'all',
            exponentformat = 'power'
        )
    )

    img = cluster.dat.img.copy()
    bestmod = cluster.model_best_fit_image.copy()
    exp = cluster.dat.exposure.copy()
    ra, dec = cluster.wcs.all_pix2world(*np.indices(cluster.dat.img.shape), 0)

    ra = ra[:, ra.shape[1] // 2]
    dec = dec[dec.shape[0] // 2, :]

    bestmod[exp == 0] = np.nan

    normalized_img = np.log10(img/bestmod)
    normalized_img[np.isnan(normalized_img)] = 0
    normalized_img[np.isinf(normalized_img)] = 0

    normalized_img[np.isnan(normalized_img)]

    region_var = np.zeros_like(img)
    region_var[cluster.psc.region_var] = 1.
    #normalized_img[exp == 0] = np.nan

    fig.add_trace(
        go.Heatmap(x=ra,
                   y=dec,
                   z=normalized_img,
                   hoverongaps=False,
                   name="img",
                   colorbar_title='Count',
                   text=['Count Map'],
                   colorscale='Picnic', #'RdBu_r'
                   zmin=np.nanmin(normalized_img),
                   zmax=-np.nanmin(normalized_img),
                   colorbar_x=.995,
                   ), row=1, col=2)

    #fig.add_trace(
    #    go.Heatmap(x=ra,
    #               y=dec,
    #               z=region_var,
    #               hoverongaps=False,
    #               name="var_region",
    #               text=['Var region'],
    #               colorscale='PiYG',
    #               opacity=0.1,
    #               hoverinfo='skip',
    #               showscale=False
    #               ), row=1, col=2)

    ymin, ymax, xmin, xmax = cluster.psc.rectangle_conv
    if xmax==cluster.dat.axes[0]: xmax -=1
    if ymax == cluster.dat.axes[1]: ymax -= 1

    fig.add_shape(type="rect",
                  x0=ra[xmin], y0=dec[ymin], x1=ra[xmax], y1=dec[ymax],
                  line=dict(color="RoyalBlue"),
                  row=1, col=2
                  )

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.55
    ))

    fig.update_xaxes(  # showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=1, col=2)
    fig.update_yaxes(  # showgrid=False,
        zeroline=False,
        showticklabels=False,
        row=1, col=2)

    fig.update_layout(
        height=800,
    )

    fig.update_xaxes(range=[ra[0] - (ra[-1]-ra[0])/100,ra[-1] + (ra[-1]-ra[0])/100], type='linear', row=1, col=2)
    fig.update_yaxes(range=[dec[0] - (dec[-1]-dec[0])/100,dec[-1] + (dec[-1]-dec[0])/100], type='linear', row=1, col=2)

    #fig.write_html("profile.html", include_mathjax='cdn')
    #fig.add_trace(go.Table(header=dict(values=list(cluster.mod.parnames)),
    #                        cells=dict(values=list(np.round(cluster.mod.params,decimals=2)))),
    #                      row=2,col=2)

    return plot(fig, include_mathjax='cdn', output_type='div')

#%%
def profile_table(cluster):

    if cluster.mod is not None:

        fig = go.Figure(data=[go.Table(header=dict(values=['Ellipse angle', 'Ellipse ratio', *cluster.mod.parnames]),
                                   cells=dict(values=[round(cluster.prof.ellangle,2), round(cluster.prof.ellratio,4), *np.round(cluster.model_best_fit, decimals=4)]))
                          ])

    else:

        fig = go.Figure(data=[go.Table(header=dict(values=['Ellipse angle', 'Ellipse ratio']),
                                       cells=dict(
                                           values=[round(cluster.prof.ellangle, 2), round(cluster.prof.ellratio, 4),
                                                   ]))
                              ])
    fig.update_layout(
        height=250
    )
    return plot(fig, include_mathjax='cdn', output_type='div')

#%%
def ps_table(cluster):
    fig = go.Figure(data=[go.Table(header=dict(values=['k_injection', 'norm', 'alpha']),
                                   cells=dict(values=[*cluster.res]))
                          ])
    fig.update_layout(
        height=250
    )

    return plot(fig, include_mathjax='cdn', output_type='div')

#%%
def header(cluster):
    fig = go.Figure(data=[go.Table(header=dict(values=['Name', 'R500 (arcmin)', 'R500 (kpc)', 'Redshift', 'Tag']),
                                   cells=dict(values=[cluster.name, round(cluster.r500_arcmin,3), cluster.r500, round(cluster.z,4), cluster.sample]))
                          ])
    fig.update_layout(
        height=250
    )
    return plot(fig, include_mathjax='cdn', output_type='div')

#%%

def dashboard(cluster, outfile='multi_plot.html'):
    
    hdr = header(cluster)
    div2 = images_expo(cluster)
    div3 = profile(cluster)
    div4 = profile_table(cluster)
    div5 = power_spectrum(cluster)
    #div6 = ps_table(cluster)

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
        {}
        </body>
    </html>
    """.format(hdr, div2, div3, div4, div5)

    with open(outfile, 'w') as f:
        f.write(html)

#%%

def analysis_header(analysis):

    fig = go.Figure(data=[go.Table(header=dict(values=['Name', *analysis.config.keys()]),
                                   cells=dict(values=[analysis.name, *analysis.config.values()]))
                          ])
    fig.update_layout(
        height=250
    )
    return plot(fig, include_mathjax='cdn', output_type='div')

def analysis_spectra(analysis):

    fig = go.Figure()

    for row in analysis.master_table:

        fig.add_scatter(x=row['k'], y=row['ps'], name=row['NAME'],
                        legendgroup=row['NAME'],
                        line_color='#636EFA',
                        error_y=dict(
                            type='data',  # value of error bar given in data coordinates
                            array=np.diag(row['ps_covariance']) ** (1 / 2),
                            visible=True))


        fig.add_scatter(x=row['k'], y=row['ps_noise'], name='{} ps_noise'.format(row['NAME']),
                        legendgroup=row['NAME'],
                        line_color='#EF553B')


    fig.update_xaxes(type="log",
                     title_text=r'$k [\text{kpc}^{-1}]$')
    fig.update_yaxes(type="log",
                     title_text=r'$\text{Power Spectrum (2D) } [\text{kpc}^{4}]$',
                     automargin=True)

    return plot(fig, include_mathjax='cdn', output_type='div')

def report(analysis):

    hdr = analysis_header(analysis)
    div2 = analysis_spectra(analysis)

    html = """\
    <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
        {}
        {}
        </body>
    </html>
    """.format(hdr, div2)

    with open(os.path.join(analysis.analysis_path, 'report.html'), 'w') as f:
        f.write(html)

#%%
def header_compare(*analysis_list):

    parameters = list(analysis_list[0].config.keys())
    cells_list = []

    for param in parameters:
        cells_list.append([analysis.config[param] for analysis in analysis_list])

    fig = go.Figure(data=[go.Table(header=dict(values=['Name', *parameters]),
                                   cells=dict(values=[[analysis.name for analysis in analysis_list],
                                                *cells_list]))])

    return plot(fig, include_mathjax='cdn', output_type='div')

def ps_compare(*analysis_list):

    fig = go.Figure()

    raw_symbols = SymbolValidator().values
    symbols=['circle', 'square', 'cross', 'star-triangle-up', 'star', 'hexagon']
    symbols = cycle(symbols)
    for analysis in analysis_list:
        symbol = next(symbols)
        colors = cycle(px.colors.qualitative.Plotly)
        for row in analysis.master_table:

            color = next(colors)
            fig.add_scatter(x=row['k']*row['R500'], y=row['ps'], name=row['NAME']+' '+analysis.name,
                        legendgroup=row['NAME'],
                        line_color=color,
                        line_width=2,
                        line_dash='dot',
                        marker_color=color,
                        marker_symbol=symbol,
                        marker_size=7,
                        opacity=0.8,
                        marker_line_color="black",
                        marker_line_width=2,
                        error_y=dict(
                            type='data',  # value of error bar given in data coordinates
                            array=np.diag(row['ps_covariance']) ** (1 / 2),
                            visible=True))

            fig.add_scatter(x=row['k'] * row['R500'], y=row['ps_noise'], name=row['NAME'] + ' PS_noise ' + analysis.name,
                            legendgroup=row['NAME'],
                            line_color='gray',
                            line_width=2,
                            line_dash='dash',
                            opacity=0.2,
                            error_y=dict(
                                type='data',  # value of error bar given in data coordinates
                                array=np.diag(row['ps_noise_covariance']) ** (1 / 2),
                                visible=True))

    fig.update_xaxes(type="log",
                     title_text=r'$k/k_{500}$')
    fig.update_yaxes(type="log",
                     title_text=r'$\text{Power Spectrum (2D) } [\text{kpc}^{4}]$',
                     automargin=True)

    return plot(fig, include_mathjax='cdn', output_type='div')

def compare(*analysis_list):

    hdr = header_compare(*analysis_list)
    div2 = ps_compare(*analysis_list)
    html = """\
        <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
            {}
            {}
            </body>
        </html>
        """.format(div2, hdr)

    with open(os.path.join('analysis_results', 'compare.html'), 'w') as f:
        f.write(html)