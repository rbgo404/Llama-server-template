INPUT_SCHEMA = {
    "image": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://github.com/rbgo404/Files/raw/main/unwant_person_clean.jpg"]
    },
    "mask": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://github.com/rbgo404/Files/raw/main/mask_unwant_person_clean.jpg"]
    },
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Remove the girl"]
    },
    "negativePrompt": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["blurry, low quality"]
    },
    "ldmSteps": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [50]
    },
    "ldmSampler": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["plms"]
    },
    "zitsWireframe": {
        'datatype': 'BOOL',
        'required': False,
        'shape': [1],
        'example': [True]
    },
    "hdStrategy": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["Resize"]
    },
    "hdStrategyCropMargin": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [32]
    },
    "hdStrategyCropTrigerSize": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [2048]
    },
    "hdStrategyResizeLimit": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [2048]
    },
    "useCroper": {
        'datatype': 'BOOL',
        'required': False,
        'shape': [1],
        'example': [False]
    },
    "croperX": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [0]
    },
    "croperY": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [0]
    },
    "croperHeight": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [512]
    },
    "croperWidth": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [512]
    },
    "sdScale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [1.0]
    },
    "sdMaskBlur": {
        'datatype': 'INT8',
        'required': False,
        'shape': [1],
        'example': [4]
    },
    "sdStrength": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [0.75]
    },
    "sdSteps": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [50]
    },
    "sdGuidanceScale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [7.5]
    },
    "sdSampler": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["uni_pc"]
    },
    "sdSeed": {
        'datatype': 'INT32',
        'required': False,
        'shape': [1],
        'example': [42]
    },
    "sdMatchHistograms": {
        'datatype': 'BOOL',
        'required': False,
        'shape': [1],
        'example': [False]
    },
    "cv2Flag": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["INPAINT_NS"]
    },
    "cv2Radius": {
        'datatype': 'INT8',
        'required': False,
        'shape': [1],
        'example': [4]
    },
    "paintByExampleSteps": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [50]
    },
    "paintByExampleGuidanceScale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [7.5]
    },
    "paintByExampleMaskBlur": {
        'datatype': 'INT8',
        'required': False,
        'shape': [1],
        'example': [0]
    },
    "paintByExampleSeed": {
        'datatype': 'INT32',
        'required': False,
        'shape': [1],
        'example': [42]
    },
    "paintByExampleMatchHistograms": {
        'datatype': 'BOOL',
        'required': False,
        'shape': [1],
        'example': [False]
    },
    "p2pSteps": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [50]
    },
    "p2pImageGuidanceScale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [1.5]
    },
    "p2pGuidanceScale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [7.5]
    },
    "controlnet_conditioning_scale": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [0.4]
    },
    "controlnet_method": {
        'datatype': 'STRING',
        'required': False,
        'shape': [1],
        'example': ["control_v11p_sd15_canny"]
    }
}