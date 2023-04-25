import dash
from dash import html, dcc, dash_table
from dash.dash_table.Format import Scheme, Trim, Format
import dash_bootstrap_components as dbc
from flask import Flask
import plotly.graph_objects as go
import os
import base64
import pandas as pd
from PIL import Image
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, State
from dash_extensions import Download
from dash_extensions.snippets import send_file
from dash_extensions.snippets import send_bytes

from skimage import data
from core.predict_system_test import predict_system
from core.utils.pdf_utils import pdf2image

from skimage import io
import plotly.express as px

server = Flask(__name__)
app = DashProxy(prevent_initial_callbacks=True, suppress_callback_exceptions=True,
                title="PDF图纸解析演示平台", server=server, transforms=[MultiplexerTransform()])
app._favicon = "logo.png"

navbar = dbc.Navbar(
    [
        dbc.Container(
            [
                html.Div(
                    [
                        html.Div([
                            dbc.NavbarBrand("PDF图纸解析演示平台",
                                            style={"fontSize": "32px", "fontWeight": "bold", "letterSpacing": "2px",
                                                   "paddingLeft": "12px"})
                        ])
                    ], style={"display": "flex", "alignItems": "center"}
                ),
                html.Div(
                    [
                        dbc.Button('截图', id="shot-png", n_clicks=0),
                    ]
                ),
                html.Div(
                    [
                        dcc.Upload(dbc.Button('上传PDF图纸'), id="upload-pdf")
                    ]
                )
            ])
    ],
    color="dark",
    dark=True,
    className="page-header")

table = dash_table.DataTable(
    id="random-data-table",
    style_data={
        'whiteSpace': 'normal',
        'height': 'auto',
    },
    style_table={'overflow': 'auto'},
    style_cell={
        "textAlign": "left"
    },
    style_cell_conditional=[
        {
            'if': {'column_id': 'scores'},
            "width": "80px",
            "minWidth": "80px",
        }
    ],
    style_data_conditional=[
        {
            "if": {
                "column_id": "scores",
                "filter_query": "{scores}>=0.8 && {scores}<0.9",
            },
            "backgroundColor": "yellow",
            "fontWeight": "bold"
        }, {
            "if": {
                "column_id": "scores",
                "filter_query": "{scores}<0.8 && {scores}>=0",
            },
            "backgroundColor": "red",
            "fontWeight": "bold"
        }
    ],
    columns=[
        {
            "id": "texts",
            "name": "文本",
        }, {
            "id": "scores",
            "name": "置信度",
            "type": "numeric",
            "format": Format(precision=4, scheme=Scheme.fixed, trim=Trim.yes)
        }, {
            "id": "lower",
            "name": "下公差限",
        },
        {
            "id": "notion",
            "name": "名义值",
        },
        {
            "id": "upper",
            "name": "上公差限",
        },
        {
            "id": "semantics",
            "name": "语义信息",
        }
    ],
    style_as_list_view=False,
    style_header={"fontWeight": "bold"}
)

fig = go.Figure()
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)
fig.update_layout(
    margin={"l": 0, "r": 0, "t": 0, "b": 0}
)
fig.update_layout(dragmode="drawrect", )
graph = dcc.Graph(id="pdf-image",
                  config={"modeBarButtonsToAdd": [
                      "drawline",
                      "drawopenpath",
                      "drawclosedpath",
                      "drawcircle",
                      "drawrect",
                      "eraseshape",
                  ]
                  },
                  style={"height": "100%", "width": "100%"},
                  responsive=True,
                  figure=fig)

content = html.Div(
    id="page-content",
    className="page-content",
    children=[
        dcc.Loading(
            [
                dbc.Card(
                    [
                        dbc.CardHeader([html.H6("PDF图纸", id="pdf-name"), html.H6("PNG截图", id="png-name"),
                                        dcc.Upload(dbc.Button('上传PNG图纸'), id="upload-png"),
                                        dbc.Button('上传该截图', id="upload-shot-png")],
                                       style={"display": "flex", "justifyContent": "space-between"}),
                        dbc.CardBody(
                            graph
                        )
                    ],
                    style={"flex": "1", "marginRight": "16px"}
                )],
            parent_style={"display": "flex", "flex": 1}
        ),
        dbc.Card(
            [
                dbc.CardHeader([html.H6("解析结果"),
                                dbc.Button("下载", id="btn-download",
                                           style={"padding": "0 8px"}),
                                Download(id="download-excel")
                                ],
                               style={"display": "flex", "justifyContent": "space-between"}),
                dbc.CardBody(
                    [
                        dcc.Loading([table])
                    ]
                )
            ],
            style={"flex": "0.3"}
        )
    ]
)

app.layout = html.Div(
    [
        dbc.Toast(
            id="toast",
            header="提示",
            is_open=False,
            dismissable=True,
            duration=3000,
            icon="danger",
            style={"position": "fixed", "top": 54,
                   "right": 10, "maxWidth": 350, "zIndex": 9999},
        ),
        navbar,
        content,
        html.Div([
            html.H6("© 2020-2021 苏州领质信息技术有限公司 - 版权所有",
                    className="page-footer-text")
        ], className="page-footer")
    ],
    className="page-container")


@app.callback(
    Output("download-excel", "data"),
    Input("btn-download", "n_clicks"),
    State("pdf-name", "children")
)
def download_excel(n_clicks, pdf_name):
    if n_clicks and ".pdf" in pdf_name:
        name = pdf_name.replace("PDF图纸：", "").replace(".pdf", "")
        return send_file(os.path.join("Uploads", name, name + "_0_out.xlsx"), filename=name + ".xlsx")


@app.callback(
    [
        Output('pdf-name', 'children'),
        Output("pdf-image", "figure"),
        Output("toast", "children"),
        Output("toast", "is_open")
    ],
    Input('upload-pdf', 'contents'),
    [
        State('upload-pdf', 'filename')
    ])
def on_click_upload2(contents, file_name):
    if contents is not None and file_name is not None:
        folder_name = file_name[: -4]
        save_dir = os.path.join("Uploads", folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if file_name[-4:] == ".pdf":
            d = contents.encode("utf8").split(b";base64,")[1]
            with open(os.path.join(save_dir, file_name), "wb") as fp:
                fp.write(base64.decodebytes(d))

            pdf2image(save_dir, folder_name, 8, 8, 0)
            img = Image.open(os.path.join(
                save_dir, folder_name + "_0_input.png"))
            scale_factor = 1
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[0, img.width * scale_factor],
                    y=[0, img.height * scale_factor],
                    mode="markers",
                    marker_opacity=0
                )
            )

            fig.update_xaxes(visible=False, range=[
                0, img.width * scale_factor])
            fig.update_yaxes(visible=False, range=[
                0, img.height * scale_factor], scaleanchor="x")
            fig.add_layout_image(
                dict(
                    x=0,
                    sizex=img.width * scale_factor,
                    y=img.height * scale_factor,
                    sizey=img.height * scale_factor,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    sizing="contain",
                    layer="below",
                    source=img
                )
            )
            fig.update_layout(
                autosize=False,
                width=img.width * scale_factor,
                height=img.height * scale_factor,
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                dragmode="drawrect"
            )

            return "PDF图纸：" + file_name, fig, "", False
        else:
            return dash.no_update, dash.no_update, "请上传PDF文件！", True


@app.callback(
    [
        Output("random-data-table", "data")
    ],
    Input("pdf-name", "children"))
def on_click_upload(children):
    if children != "PDF图纸：":
        folder_name = children[6:-4]
        save_dir = os.path.join("Uploads", folder_name)
        result = predict_system(save_dir, folder_name, True)
        return pd.DataFrame(
            {"texts": result[0],
             "scores": result[2],
             "locations": result[1],
             'lower': result[3],
             'notion': result[4],
             'upper': result[5],
             "semantics": result[6]}).to_dict("records")
    else:
        return dash.no_update


@app.callback(
    Output("pdf-image", "figure"),
    Input("random-data-table", "active_cell"),
    [
        State("random-data-table", "data"),
        State("pdf-image", "figure")
    ])
def on_click_upload(active_cell, data, figure):
    locations = data[active_cell["row"]]["locations"]
    height = figure["layout"]["images"][0]["y"]
    figure["layout"]["shapes"] = [
        {
            "type": 'rect',
            "xref": 'x',
            "yref": 'y',
            "x0": locations[3][0],
            "y0": height - locations[3][1],
            "x1": locations[2][0],
            "y1": height - locations[0][1],
            "opacity": 0.2,
            "fillcolor": 'red',
            "line": {
                "color": 'red'
            }
        }
    ]

    return figure


# 修改尝试
@app.callback(
    [
        Output('png-name', 'children'),
        Output("png-image", "figure")
    ],
    Input('upload-png', 'contents'),
    [
        State('upload-png', 'filename')
    ])
def on_click_upload3(contents, file_name):
    if contents is not None and file_name is not None:
        folder_name = file_name[: -4]
        save_dir = os.path.join("Uploads", folder_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 这里写的简单了，还得修改。
        if file_name[-4:] == ".png":
            real_name = folder_name + "_0_input.png"
            d = contents.encode("utf8").split(b";base64,")[1]
            with open(os.path.join(save_dir, real_name), "wb") as fp:
                fp.write(base64.decodebytes(d))

            img = Image.open(os.path.join(
                save_dir, folder_name + "_0_input.png"))
            scale_factor = 1
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[0, img.width * scale_factor],
                    y=[0, img.height * scale_factor],
                    mode="markers",
                    marker_opacity=0
                )
            )

            fig.update_xaxes(visible=False, range=[
                0, img.width * scale_factor])
            fig.update_yaxes(visible=False, range=[
                0, img.height * scale_factor], scaleanchor="x")
            fig.add_layout_image(
                dict(
                    x=0,
                    sizex=img.width * scale_factor,
                    y=img.height * scale_factor,
                    sizey=img.height * scale_factor,
                    xref="x",
                    yref="y",
                    opacity=1.0,
                    sizing="contain",
                    layer="below",
                    source=img
                )
            )
            fig.update_layout(
                autosize=False,
                width=img.width * scale_factor,
                height=img.height * scale_factor,
                margin={"l": 0, "r": 0, "t": 0, "b": 0}
            )

            return "Png截图：" + file_name, fig
        else:
            return dash.no_update, dash.no_update


@app.callback(
    [
        Output("random-data-table", "data")
    ],
    Input("png-name", "children"))
def on_click_upload4(children):
    if children != "Png截图：":
        folder_name = children[6:-4]
        save_dir = os.path.join("Uploads", folder_name)
        result = predict_system(save_dir, folder_name, True)
        return pd.DataFrame(
            {"texts": result[0],
             "scores": result[2],
             "locations": result[1],
             'lower': result[3],
             'notion': result[4],
             'upper': result[5],
             "semantics": result[6]}).to_dict("records")
    else:
        return dash.no_update


@app.callback(
    Output("png-image", "figure"),
    Input("random-data-table", "active_cell"),
    [
        State("random-data-table", "data"),
        State("png-image", "figure")
    ])
def on_click_upload5(active_cell, data, figure):
    locations = data[active_cell["row"]]["locations"]
    height = figure["layout"]["images"][0]["y"]
    figure["layout"]["shapes"] = [
        {
            "type": 'rect',
            "xref": 'x',
            "yref": 'y',
            "x0": locations[3][0],
            "y0": height - locations[3][1],
            "x1": locations[2][0],
            "y1": height - locations[0][1],
            "opacity": 0.2,
            "fillcolor": 'red',
            "line": {
                "color": 'red'
            }
        }
    ]

    return figure


@app.callback(
    Output("download-excel", "data"),
    Input("btn-download", "n_clicks"),
    State("png-name", "children")
)
def download_excel2(n_clicks, png_name):
    if n_clicks and ".png" in png_name:
        name = png_name.replace("Png截图：", "").replace(".png", "")
        return send_file(os.path.join("Uploads", name, name + "_0_out.xlsx"), filename=name + ".xlsx")


@app.callback(
    Output("pdf-image", "figure"),
    Input("shot-png", "n_clicks"),
    State('upload-pdf', 'filename')
)
def on_click_upload6(n_clicks, file_name):
    if n_clicks:
        folder_name = file_name[: -4]
        save_dir = os.path.join("Uploads", folder_name)
        img = io.imread(os.path.join(
            save_dir, folder_name + "_0_input.png"))
        fig = px.imshow(img, binary_string=True)
        fig.update_layout(dragmode="drawrect", )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        config = {
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ]
        }
        return fig


@app.callback(
    Output("pdf-image", "figure"),
    Input("pdf-image", "relayoutData"),
    State('upload-pdf', 'filename')
)
def on_new_annotation(relayout_data, file_name):
    if "shapes" in relayout_data:
        last_shape = relayout_data["shapes"][-1]
        folder_name = file_name[: -4]
        save_dir = os.path.join("Uploads", folder_name)
        # shape coordinates are floats, we need to convert to ints for slicing
        x0, y0 = int(last_shape["x0"]), int(last_shape["y0"])
        x1, y1 = int(last_shape["x1"]), int(last_shape["y1"])
        img = io.imread(os.path.join(
            save_dir, folder_name + "_0_input.png"))
        roi_img = img[y0:y1, x0:x1]
        io.imsave(os.path.join(
            save_dir, folder_name + '_0_shot.png'), roi_img)
        figure_new = px.imshow(roi_img, binary_string=True)
        figure_new.update_xaxes(visible=False)
        figure_new.update_yaxes(visible=False, )
        return figure_new
    else:
        return dash.no_update


@app.callback(
    Output("png-name", "children"),
    Output("png-image", "figure"),
    Input("upload-shot-png", "n_clicks"),
    State('upload-pdf', 'filename')
)
def on_click_upload8(n_clicks, file_name):
    if n_clicks and file_name is not None:
        folder_name = file_name[: -4]
        save_name = folder_name + '_shot'
        save_dir = os.path.join("Uploads", save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if file_name[-4:] == ".pdf":
            real_name = save_name + "_0_input.png"
        save_dir2 = os.path.join("Uploads", folder_name)
        img2 = io.imread(os.path.join(save_dir2, folder_name + '_0_shot.png'))
        io.imsave(os.path.join(save_dir, save_name + "_0_input.png"), img2)
        img = io.imread(os.path.join(
            save_dir, save_name + "_0_input.png"))
        output_name = folder_name + '_shot' + '.png'
        figure_new = px.imshow(img, binary_string=True)
        figure_new.update_xaxes(visible=False)
        figure_new.update_yaxes(visible=False, )
        return "Png截图：" + output_name, figure_new
    else:
        return dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False)