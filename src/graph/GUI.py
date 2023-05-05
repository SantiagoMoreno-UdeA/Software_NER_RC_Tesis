# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 16:01:08 2022

@author: Santiago Moreno
"""

import os 
import gradio as gr
import sys
import json


default_path = os.path.dirname(os.path.abspath(__file__))
#default_path = default_path.replace('\\', '/')

os.chdir(default_path)
sys.path.insert(0, default_path+'/../scripts')

from src.scripts.functionsner import use_model, tag_sentence, json_to_txt, training_model, characterize_data, upsampling_data, usage_cuda, copy_data
from src.scripts.functionsrc import  use_model_rc, training_model_rc, usage_cuda_rc 

models = os.listdir(default_path+'/../../models')
models.remove('RC')
models_rc = os.listdir(default_path+'/../../models/RC')

#-------------------------------------------Functions-----------------------------------------------

#--------------------------------------NER-----------------------------------
def Trainer(fast, model_name, standard, input_dir, Upsampling, Cuda):
    if fast: epochs = 1
    else: epochs = 20
    
    if Cuda: 
        cuda_info = usage_cuda(True)
    else: 
        cuda_info = usage_cuda(False)
    
    
    if standard:
        copy_data(input_dir)
    else:
        Error = json_to_txt(input_dir)
        if type(Error)==int:
            yield 'Error processing the input documents, code error {}'.format(Error)
    if Upsampling:
        yield cuda_info+'\n'+'-'*20+'Upsampling'+'-'*20
        entities_dict=characterize_data()
        entities = list(entities_dict.keys())
        entities_to_upsample = [entities[i] for i,value in enumerate(entities_dict.values()) if value < 200]
        upsampling_data(entities_to_upsample, 0.8,  entities)
        yield '-'*20+'Training'+'-'*20
    else:
        yield cuda_info+'\n'+'-'*20+'Training'+'-'*20
    Error = training_model(model_name, epochs)
    if type(Error)==int:
        yield 'Error training the model, code error {}'.format(Error)
    else: 
        yield 'Training complete, model {} could be found at models/{}'.format(model_name,model_name)


def Tagger_sentence(Model, Sentence, Cuda):
    if Cuda: cuda_info = usage_cuda(True)
    else: cuda_info = usage_cuda(False)
    yield cuda_info+'\n'+'-'*20+'Tagging'+'-'*20
    results = tag_sentence(Sentence, Model)
    if type(results)==int:
        yield "Error {}, see documentation".format(results)
    else:
        yield results['Highligth']

def Tagger_json(Model, Input_file, Output_file, Cuda):
    if Cuda: cuda_info = usage_cuda(True)
    else: cuda_info = usage_cuda(False)
    
    with open(Output_file, "w", encoding='utf-8') as write_file:
        json.dump({'error':'error'}, write_file)
        
    yield cuda_info+'\n'+'-'*20+'Tagging'+'-'*20, {}, Output_file
    
    results = use_model(Model, Input_file.name, Output_file)
    if type(results)==int:
        error_dict = {}
        yield "Error {}, see documentation".format(results), error_dict, Output_file
    else:
        yield { "text" : results['text'], 'entities': results['entities']}, results, Output_file


#--------------------RC-------------------------------
def Trainer_RC(fast, model_name, input_file, rel2id_file, Cuda):
    if fast: epochs = 1
    else: epochs = 200
    
    if Cuda: 
        cuda_info = usage_cuda_rc(True)
    else: 
        cuda_info = usage_cuda_rc(False)
    

    yield cuda_info+'\n'+'-'*20+'Training'+'-'*20
    Error = training_model_rc(model_name, input_file.name, rel2id_file.name ,epochs)
    if type(Error)==int:
        yield 'Error training the model, code error {}'.format(Error)
    else: 
        yield 'Training complete, model {} could be found at models/{}'.format(model_name,model_name)


def Tagger_document_RC(Model, Input_file, Output_file, Cuda):
    if Cuda: cuda_info = usage_cuda_rc(True)
    else: cuda_info = usage_cuda_rc(False)
    
    with open(Output_file, "w", encoding='utf-8') as write_file:
        json.dump({'error':'error'}, write_file)
        
    yield {'cuda':cuda_info}, Output_file
    
    results = use_model_rc(Model, Input_file.name, Output_file)
    if type(results)==int:
        error_dict = {}
        yield  error_dict, Output_file
    else:
        yield results, Output_file
        
        
#---------------------------------GUI-------------------------------------
def execute_GUI():
    global models
    with gr.Blocks(title='NER', css="#title {font-size: 150% } #sub {font-size: 120% } ") as demo:
        
        gr.Markdown("Named Entity Recognition(NER) and Relation Classification (RC) by GITA and PRATECH.",elem_id="title")
        gr.Markdown("Software developed by Santiago Moreno, Juan Camilo Vasquez, Daniel Escobar, and Rafael Orozco",elem_id="sub")
        gr.Markdown("Named Entity Recognition(NER) and Relation Classification (RC) System.")

        with gr.Tab("NER"):
            gr.Markdown("Use Tagger to apply NER from a pretrained model in a sentence or a given document in INPUT (.JSON) format.")
            gr.Markdown("Use Trainer to train a new NER model from a directory of documents in PRATECH (.JSON) format.")
            with gr.Tab("Tagger"):
                with gr.Tab("Sentence"):
                    with gr.Row():
                        with gr.Column():
                            b = gr.Radio(list(models), label='Model')
                            inputs =[
                                 b,
                                 gr.Textbox(placeholder="Enter sentence here...", label='Sentence'), 
                                 gr.Radio([True,False], label='CUDA', value=False),
                            ]
                            tagger_sen = gr.Button("Tag")
                        output = gr.HighlightedText()
                    
               
                    
                    tagger_sen.click(Tagger_sentence, inputs=inputs, outputs=output)
                    b.change(fn=lambda value: gr.update(choices=list(os.listdir('../../models'))), inputs=b, outputs=b)
                    gr.Examples(
                    
                        examples=[
                            ['CCC',"Camara de comercio de medellín. El ciudadano JAIME JARAMILLO VELEZ identificado con C.C. 12546987 ingresó al plantel el día 1/01/2022"],
                            ['CCC',"Razón Social GASEOSAS GLACIAR S.A.S, ACTIVIDAD PRINCIPAL fabricación y distribución de bebidas endulzadas"]
                         ],
                        inputs=inputs
                        )
          
                   
                with gr.Tab("Document"):
                    with gr.Row():
                        with gr.Column(): 
                            c = gr.Radio(list(models), label='Model')
                            inputs =[
                                 c,
                                 gr.File(label='Input data file'),
                                 gr.Textbox(placeholder="Enter path here...", label='Output data file path'), #value='../../data/Tagged/document_tagged.json'),
                                 gr.Radio([True,False], label='CUDA', value=False),
                            ]
                            tagger_json = gr.Button("Tag")
                        output = [
                            gr.HighlightedText(),
                            gr.JSON(),
                            gr.File(),
                            ]
                        
                    models = os.listdir(default_path+'/../../models')
                    models.remove('RC')
                    
                    tagger_json.click(Tagger_json, inputs=inputs, outputs=output)
                    c.change(fn=lambda value: gr.update(choices=list(os.listdir('../../models')).remove('RC')), inputs=c, outputs=c)
                    
             
            with gr.Tab("Trainer"):
                with gr.Row():
                    with gr.Column():
                        train_input = inputs =[
                             gr.Radio([True,False], label='Fast training', value=True),
                             gr.Textbox(placeholder="Enter model name here...", label='New model name'),
                             gr.Radio([True,False], label='Standard input', value=False),
                             gr.Textbox(placeholder="Enter path here...", label='Input data directory path'), 
                             gr.Radio([True,False], label='Upsampling', value=False),
                             gr.Radio([True,False], label='CUDA', value=False),
                        ]
                        trainer = gr.Button("Train")
                    train_output = gr.TextArea(placeholder="Output information", label='Output')
                    
                    
        with gr.Tab("RC"):
            gr.Markdown("Use Tagger to apply RC from a pretrained model in document in  (.TXT) CONLL04 format.")
            gr.Markdown("Use Trainer to train a new RC model from a  file (.TXT) CONLL04 format and the rel2id file (.JSON).")
            with gr.Tab("Tagger Document"):

                with gr.Row():
                    with gr.Column(): 
                        c = gr.Radio(list(models_rc), label='Model')
                        inputs =[
                             c,
                             gr.File(label='Input data file'),
                             gr.Textbox(placeholder="Enter path here...", label='Output data file path (.JSON)'), #value='../../data/Tagged/document_tagged.json'),
                             gr.Radio([True,False], label='CUDA', value=False),
                        ]
                        tagger_json = gr.Button("Tag")
                    output = [
                        gr.JSON(),
                        gr.File(),
                        ]
                
                tagger_json.click(Tagger_document_RC, inputs=inputs, outputs=output)
                c.change(fn=lambda value: gr.update(choices=list(os.listdir('../../models/RC'))), inputs=c, outputs=c)

            with gr.Tab("Trainer"):
                with gr.Row():
                    with gr.Column():
                        train_input = inputs =[
                             gr.Radio([True,False], label='Fast training', value=True),
                             gr.Textbox(placeholder="Enter model name here...", label='New model name'),
                             gr.File(label='Input train file (.TXT)'),
                             gr.File(label='Input rel2id file (.JSON)'), 
                             gr.Radio([True,False], label='CUDA', value=False),
                        ]
                        trainer = gr.Button("Train")
                    train_output = gr.TextArea(placeholder="Output information", label='Output')
                    
        trainer.click(Trainer_RC, inputs=train_input, outputs=train_output)
        

        
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=8080,inbrowser=True)


