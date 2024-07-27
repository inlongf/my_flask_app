from dotenv import load_dotenv
import os
import threading
import datetime
import torch
from flask import Flask, request, jsonify, session, render_template, redirect, url_for, send_file, Response, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from werkzeug.security import generate_password_hash, check_password_hash
from forms import LoginForm
from flask_migrate import Migrate
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from model import (
    BERT_LSTM_Model, GCN_GRU_Model, GANGenerator, GANDiscriminator, 
    GRU_VAE, LSTM_PolicyNetwork, AdvancedMultimodalTransformer,
    SelfSupervisedReinforcementLearningModel, MixtureOfExpertsModel,
    MetaLearningModel, MemoryAugmentedNetwork, FineTunedGPTModel,
    HierarchicalReinforcementLearningModel, SelfSupervisedRecursiveNN,
    Column, ThousandBrainsModel, AffectiveModel, MetaPolicyNetwork,
    MetaReinforcementLearning, EnhancedMemoryNetwork, SocialInteractionModel,
    NeuralTuringMachine, SNN_LSTM_Model, db, User, Conversation, InternetAccessRequest
)
from config import device
from utils import (
    generate_caption, send_message_to_kafka, read_message_from_kafka, fuse_sensor_data, generate_stable_diffusion_image,
    generate_falcon_text, continual_learning_model, meta_learning_model, adversarial_training, sign_command,
    verify_command, users, online_learning, validate_code, apply_code_update, parse_natural_language_command,
    fetch_web_content, recognize_speech, synthesize_speech, recognize_faces_in_image, recognize_iris_in_image,
    perform_satellite_detection, provide_real_time_navigation, detect_objects_yolo, search_similar_features, 
    kafka_consumer_thread, process_large_dataset, generate_synthetic_data, load_model, deploy_model_to_edge,
    detect_network_threats, ensure_model_fairness, explain_model_predictions, climate_change_analysis,
    create_creative_content, control_embodied_agent, gen_video
)
import unicodedata

load_dotenv()  # 加载 .env 文件

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://newuser:money456742@localhost/chatbot'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
csrf = CSRFProtect(app)
migrate = Migrate(app, db)

# Configure logging
if not os.path.exists('logs'):
    os.mkdir('logs')
log_path = 'logs/app.log'
handler = RotatingFileHandler(log_path, maxBytes=10000, backupCount=1, encoding='utf-8')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

# Set default dtype to float32
torch.set_default_dtype(torch.float32)

# Function to convert full-width characters to half-width
def to_half_width(s):
    return ''.join(unicodedata.normalize('NFKC', c) for c in s)

@app.before_request
def log_request_info():
    app.logger.info('Headers: %s', request.headers)
    app.logger.info('Body: %s', request.get_data(as_text=True))

@app.errorhandler(Exception)
def log_error_info(e):
    app.logger.error('Error: %s', str(e))
    return jsonify(error=str(e)), 500

@app.errorhandler(400)
def bad_request_error(e):
    app.logger.error('Bad Request: %s', str(e))
    return jsonify(error='Bad Request'), 400

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['username'] = username
            session['role'] = user.role
            return redirect(url_for('chat'))
        flash('用户名或密码错误')
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/chat', methods=['GET', 'POST'])
@csrf.exempt
def chat():
    if request.method == 'GET':
        if 'username' not in session:
            return redirect(url_for('login'))
        return render_template('chat.html')

    if request.method == 'POST':
        if 'username' not in session:
            return jsonify({'response': '請先登入'}), 403

        app.logger.debug('POST data: %s', request.data)

        try:
            data = request.get_json()
            app.logger.debug('Parsed JSON data: %s', data)
        except Exception as e:
            app.logger.error('Error parsing JSON: %s', e)
            return jsonify({'response': 'Invalid JSON'}), 400

        if data is None or 'input' not in data or 'type' not in data:
            app.logger.error('Invalid input data: %s', data)
            return jsonify({'response': 'Invalid input'}), 400

        user_input = to_half_width(data['input'])
        input_type = data['type']
        app.logger.debug('User input: %s', user_input)
        app.logger.debug('Input type: %s', input_type)

        response = ""
        try:
            if input_type == "text":
                if "text classification" in user_input:
                    model = BERT_LSTM_Model().to(device)
                    tokenizer = GPT2Tokenizer.from_pretrained('bert-base-uncased')
                    inputs = tokenizer.encode(user_input, return_tensors='pt').to(device)
                    outputs = model(inputs)
                    response = f"Classification result: {outputs.tolist()}"
                elif "text generation" in user_input:
                    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
                    inputs = tokenizer.encode(user_input, return_tensors='pt').to(device)
                    outputs = model.generate(
                        inputs,
                        max_length=50,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7
                    )
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                elif "emotion analysis" in user_input:
                    model = AffectiveModel().to(device)
                    tokenizer = GPT2Tokenizer.from_pretrained('bert-base-uncased')
                    inputs = tokenizer.encode(user_input, return_tensors='pt').to(device)
                    emotion_logits, _ = model(inputs)
                    response = f"Emotion analysis result: {emotion_logits.tolist()}"
                else:
                    response = "Hello! How can I assist you today?"  # Default response for general queries
            elif input_type == "image":
                if "generate image" in user_input:
                    model = GANGenerator().to(device)
                    noise = torch.randn((1, 100)).to(device)
                    generated_image = model(noise)
                    response = f"Generated image tensor: {generated_image.tolist()}"
                else:
                    response = "Invalid image query. Please specify a valid task."
            elif input_type == "multimodal":
                text_input = data.get('text')
                image_input = data.get('image')
                if text_input and image_input:
                    model = AdvancedMultimodalTransformer().to(device)
                    prediction = model(text_input, image_input)
                    response = f"Multimodal analysis result: {prediction.tolist()}"
                else:
                    response = "Multimodal query requires both text and image inputs."
            else:
                response = "Invalid input type. Please specify a valid input type."
                
        except Exception as e:
            app.logger.error('Error processing model: %s', str(e))
            return jsonify({'response': 'Error processing input'}), 500

        app.logger.debug('Model response: %s', response)

        user = User.query.filter_by(username=session['username']).first()
        conversation = Conversation(user_id=user.id, message=user_input, response=response)
        db.session.add(conversation)
        db.session.commit()

        return jsonify({'response': response})

@app.route('/voice_input', methods=['POST'])
@csrf.exempt
def voice_input():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    file = request.files['file']
    text = recognize_speech(file)
    user = User.query.filter_by(username=session['username']).first()
    conversation = Conversation(user_id=user.id, message=text, response=text)
    db.session.add(conversation)
    db.session.commit()
    return jsonify({'response': text})

@app.route('/video_input', methods=['POST'])
@csrf.exempt
def video_input():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    file = request.files['file']
    video_path = os.path.join('static', 'uploads', file.filename)
    file.save(video_path)
    response = "Video received and saved."
    user = User.query.filter_by(username=session['username']).first()
    conversation = Conversation(user_id=user.id, message='video uploaded', response=response)
    db.session.add(conversation)
    db.session.commit()
    return jsonify({'response': response})

@app.route('/request_internet_access', methods=['POST'])
@csrf.exempt
def request_internet_access():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    query = data['query']

    user = User.query.filter_by(username=session['username']).first()
    access_request = InternetAccessRequest(user_id=user.id, query=query)
    db.session.add(access_request)
    db.session.commit()

    return jsonify({'response': '申請已提交，等待批准'})

@app.route('/approve_request', methods=['POST'])
@csrf.exempt
def approve_request():
    if 'username' not in session or session['role'] != 'creator':
        return jsonify({'response': '只有創建者可以批准申請'}), 403

    data = request.get_json()
    query = data['query']
    access_request = InternetAccessRequest.query.filter_by(query=query, status='pending').first()
    if access_request:
        access_request.status = 'approved'
        db.session.commit()
        result = fetch_web_content(query)
        return jsonify({'response': f'資料已獲取: {result}'})

    return jsonify({'response': '未找到匹配的申請或申請已被處理'})

@app.route('/fetch_internet_data', methods=['POST'])
@csrf.exempt
def fetch_internet_data_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    query = data['query']
    result = fetch_web_content(query)
    return jsonify({'response': f'資料已獲取: {result}'})

@app.route('/monitor_users', methods=['GET'])
def monitor_users():
    if 'username' not in session or session['role'] != 'creator':
        return jsonify({'response': '只有創建者可以監控用戶活動'}), 403

    conversations = Conversation.query.all()
    internet_requests = InternetAccessRequest.query.all()

    conversation_data = [{'username': User.query.get(conv.user_id).username, 'message': conv.message, 'response': conv.response, 'timestamp': conv.timestamp} for conv in conversations]
    request_data = [{'username': User.query.get(req.user_id).username, 'query': req.query, 'status': req.status, 'timestamp': req.timestamp} for req in internet_requests]

    return jsonify({
        'conversations': conversation_data,
        'internet_requests': request_data
    })

@app.route('/update_code', methods=['POST'])
@csrf.exempt
def update_code():
    if 'username' not in session or session['role'] != 'creator':
        return jsonify({'response': '只有創建者可以修改代碼'}), 403

    data = request.get_json()
    code = data['code']
    if validate_code(code):
        success = apply_code_update(code)
        if success:
            return jsonify({'response': '代碼更新成功'})
        else:
            return jsonify({'response': '代碼更新失敗'}), 500
    else:
        return jsonify({'response': '代碼驗證失敗'}), 400

@app.route('/download_code', methods=['GET'])
def download_code():
    if 'username' not in session or session['role'] != 'creator':
        return jsonify({'response': '只有創建者可以下載代碼'}), 403

    return send_file('path_to_your_code_file', as_attachment=True)

@app.route('/generate', methods=['POST'])
@csrf.exempt
def generate():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    prompt = data['prompt']
    image = generate_stable_diffusion_image(prompt)
    image_path = 'static/generated_image.png'
    image.save(image_path)
    return jsonify({'image_path': image_path})

@app.route('/generate_text', methods=['POST'])
@csrf.exempt
def generate_text():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    prompt = data['prompt']
    response = process_large_dataset([prompt])
    return jsonify({'response': response})

@app.route('/multimodal', methods=['POST'])
@csrf.exempt
def multimodal():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    text_input = data['text']
    image_input = data['image']
    model = AdvancedMultimodalTransformer().to(device)
    prediction = model(text_input, image_input)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/reinforce', methods=['POST'])
@csrf.exempt
def reinforce():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    state = data['state']
    model = SelfSupervisedReinforcementLearningModel(state_dim=len(state), action_dim=4).to(device)
    action_probs, predicted_state = model(torch.tensor(state).unsqueeze(0).unsqueeze(0).to(device))
    return jsonify({'action_probs': action_probs.tolist(), 'predicted_state': predicted_state.tolist()})

@app.route('/experts', methods=['POST'])
@csrf.exempt
def experts():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    input_data = data['input']
    expert1 = Column(input_dim=len(input_data), output_dim=1).to(device)
    expert2 = Column(input_dim=len(input_data), output_dim=1).to(device)
    model = MixtureOfExpertsModel([expert1, expert2]).to(device)
    output = model(torch.tensor(input_data).to(device))
    return jsonify({'output': output.tolist()})

@app.route('/meta', methods=['POST'])
@csrf.exempt
def meta():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    input_data = data['input']
    base_model = Column(input_dim=len(input_data), output_dim=1).to(device)
    model = MetaLearningModel(base_model).to(device)
    output = model(torch.tensor(input_data).to(device))
    return jsonify({'output': output.tolist()})

@app.route('/memory', methods=['POST'])
@csrf.exempt
def memory():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    input_data = data['input']
    model = MemoryAugmentedNetwork(input_dim=len(input_data), memory_units=10, memory_unit_size=20).to(device)
    output = model(torch.tensor(input_data).unsqueeze(0).to(device))
    return jsonify({'output': output.tolist()})

@app.route('/fetch_info', methods=['POST'])
@csrf.exempt
def fetch_info():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    query = data['query']
    info = fetch_web_content(query)
    return jsonify({'information': info})

@app.route('/generate_caption', methods=['POST'])
@csrf.exempt
def generate_caption_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    image_path = data['image_path']
    caption = generate_caption(image_path)
    return jsonify({'caption': caption})

@app.route('/generate_synthetic_data', methods=['POST'])
@csrf.exempt
def generate_synthetic_data_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    images = np.array(data['images'])
    labels = np.array(data['labels'])
    synthetic_data = generate_synthetic_data(images, labels)
    return jsonify({'synthetic_data': synthetic_data})

@app.route('/deploy_to_edge', methods=['POST'])
@csrf.exempt
def deploy_to_edge_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    model_path = request.get_json()['model_path']
    edge_device_path = request.get_json()['edge_device_path']
    model = load_model(model_path).to(device)
    deploy_model_to_edge(model, edge_device_path)
    return jsonify({'status': 'Model deployed to edge device'})

@app.route('/detect_threats', methods=['POST'])
@csrf.exempt
def detect_threats_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    packet = request.get_json()['packet']
    detect_network_threats(packet)
    return jsonify({'status': 'Network threats detected'})

@app.route('/ensure_fairness', methods=['POST'])
@csrf.exempt
def ensure_fairness_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    model_path = request.get_json()['model_path']
    data = np.array(request.get_json()['data'])
    labels = np.array(request.get_json()['labels'])
    model = load_model(model_path).to(device)
    accuracy = ensure_model_fairness(model, data, labels)
    return jsonify({'fairness_accuracy': accuracy})

@app.route('/explain_predictions', methods=['POST'])
@csrf.exempt
def explain_predictions_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    model_path = request.get_json()['model_path']
    data = np.array(request.get_json()['data'])
    model = load_model(model_path).to(device)
    explain_model_predictions(model, data)
    return jsonify({'status': 'Model predictions explained'})

@app.route('/climate_analysis', methods=['POST'])
@csrf.exempt
def climate_analysis_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()['data']
    predictions = climate_change_analysis(data)
    return jsonify({'climate_predictions': predictions})

@app.route('/create_creative_content', methods=['POST'])
@csrf.exempt
def create_creative_content_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    noise = np.random.normal(0, 1, (1, 100))
    creative_content = create_creative_content(noise)
    return jsonify({'creative_content': creative_content.tolist()})

@app.route('/embodied_agent', methods=['POST'])
@csrf.exempt
def embodied_agent_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    action = request.get_json()['action']
    control_embodied_agent(action)
    return jsonify({'status': 'Action applied to embodied agent'})

@app.route('/continuous_learning', methods=['POST'])
@csrf.exempt
def continuous_learning_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    cl_strategy, benchmark = continual_learning_model()
    results = []
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        results.append(cl_strategy.eval(benchmark.test_stream))
    return jsonify({'results': results})

@app.route('/meta_learning', methods=['POST'])
@csrf.exempt
def meta_learning_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    maml, tasksets, opt = meta_learning_model()
    learner = l2l.algorithms.MAML(maml, lr=0.01, first_order=False)
    for iteration in range(10):
        opt.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        for task in tasksets:
            learner.clone()
            for step in range(1):
                train_error = learner.loss(task)
                learner.adapt(train_error)
            valid_error = learner.loss(task)
            meta_valid_error += valid_error.item()
            valid_error.backward()
        opt.step()
    return jsonify({'meta_train_error': meta_train_error, 'meta_valid_error': meta_valid_error})

@app.route('/adversarial_training', methods=['POST'])
@csrf.exempt
def adversarial_training_route():
    if 'username' not in session:
        return jsonify({'response': '請先登入'}), 403

    data = request.get_json()
    model = FineTunedGPTModel().to(device)
    inputs = torch.FloatTensor(data['inputs']).to(device)
    labels = torch.LongTensor(data['labels']).to(device)
    epsilon = 0.1
    output = adversarial_training(model, inputs, labels, epsilon)
    return jsonify({'output': output.tolist()})

@app.route('/video_feed')
def video_feed():
    return Response(gen_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    consumer_thread = threading.Thread(target=kafka_consumer_thread)
    consumer_thread.start()

    app.run(debug=True)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    main()
