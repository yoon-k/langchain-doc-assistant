"""
Flask API for Document Assistant
"""

import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from datetime import datetime
from werkzeug.utils import secure_filename

from app.agents.document_agent import DocumentAgent, create_document_agent
from app.tools.document_tools import document_store


def create_app():
    """Create and configure Flask application."""
    app = Flask(__name__,
                static_folder='../static',
                template_folder='../templates')
    CORS(app)

    # Configure upload settings
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'md', 'csv', 'xlsx'}

    # Store agents per session
    agents = {}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def get_agent(session_id: str) -> DocumentAgent:
        if session_id not in agents:
            agents[session_id] = create_document_agent()
        return agents[session_id]

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/static/<path:filename>')
    def serve_static(filename):
        return send_from_directory(app.static_folder, filename)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'document-assistant',
            'timestamp': datetime.utcnow().isoformat()
        })

    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.json
        session_id = data.get('session_id', 'default')
        message = data.get('message', '')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        try:
            agent = get_agent(session_id)
            response = agent.chat(message)

            return jsonify({
                'response': response,
                'documents': agent.get_documents(),
                'session_id': session_id
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/documents/upload', methods=['POST'])
    def upload_document():
        session_id = request.form.get('session_id', 'default')

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        try:
            filename = secure_filename(file.filename)
            content = file.read()

            agent = get_agent(session_id)
            result = agent.upload_document(content, filename)

            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/documents', methods=['GET'])
    def list_documents():
        session_id = request.args.get('session_id', 'default')
        agent = get_agent(session_id)

        return jsonify({
            'documents': agent.get_documents()
        })

    @app.route('/api/documents/<doc_id>', methods=['DELETE'])
    def delete_document(doc_id):
        session_id = request.args.get('session_id', 'default')
        agent = get_agent(session_id)

        result = agent.delete_document(doc_id)
        return jsonify(result)

    @app.route('/api/documents/<doc_id>/active', methods=['POST'])
    def set_active_document(doc_id):
        session_id = request.args.get('session_id', 'default')
        agent = get_agent(session_id)

        result = agent.set_active_document(doc_id)
        return jsonify(result)

    @app.route('/api/session/reset', methods=['POST'])
    def reset_session():
        data = request.json
        session_id = data.get('session_id', 'default')

        if session_id in agents:
            agents[session_id].reset()
            del agents[session_id]

        return jsonify({
            'status': 'reset',
            'session_id': session_id
        })

    return app


app = create_app()


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
