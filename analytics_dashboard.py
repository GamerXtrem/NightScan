"""Analytics and reporting dashboard for NightScan."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import io
import base64

try:
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from flask import Blueprint, render_template_string, jsonify, request, send_file
from flask_login import login_required, current_user
from sqlalchemy import func, and_, or_, desc
from datetime import datetime

from config import get_config
from metrics import track_request_metrics

logger = logging.getLogger(__name__)

# Create analytics blueprint
analytics_bp = Blueprint('analytics', __name__, url_prefix='/analytics')


@dataclass
class AnalyticsMetrics:
    """Container for analytics metrics."""
    total_detections: int = 0
    unique_species: int = 0
    active_sensors: int = 0
    avg_confidence: float = 0.0
    detections_today: int = 0
    detections_this_week: int = 0
    detections_this_month: int = 0
    top_species: List[Tuple[str, int]] = None
    hourly_distribution: Dict[int, int] = None
    daily_distribution: Dict[str, int] = None
    confidence_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.top_species is None:
            self.top_species = []
        if self.hourly_distribution is None:
            self.hourly_distribution = {}
        if self.daily_distribution is None:
            self.daily_distribution = {}
        if self.confidence_distribution is None:
            self.confidence_distribution = {}


class AnalyticsEngine:
    """Main analytics engine for generating insights."""
    
    def __init__(self, db):
        """Initialize analytics engine with database connection."""
        self.db = db
        self.config = get_config()
    
    def get_detection_metrics(self, days: int = 30) -> AnalyticsMetrics:
        """Get comprehensive detection metrics."""
        from web.app import Detection
        
        # Date ranges
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        
        # Basic metrics
        total_detections = Detection.query.filter(Detection.time >= start_date).count()
        unique_species = self.db.session.query(func.count(func.distinct(Detection.species))).filter(
            Detection.time >= start_date
        ).scalar() or 0
        
        # Time-based metrics
        detections_today = Detection.query.filter(Detection.time >= today_start).count()
        detections_this_week = Detection.query.filter(Detection.time >= week_start).count()
        detections_this_month = Detection.query.filter(Detection.time >= month_start).count()
        
        # Top species
        species_query = self.db.session.query(
            Detection.species, func.count(Detection.id).label('count')
        ).filter(Detection.time >= start_date).group_by(Detection.species).order_by(desc('count')).limit(10)
        
        top_species = [(species, count) for species, count in species_query.all()]
        
        # Hourly distribution
        hourly_query = self.db.session.query(
            func.extract('hour', Detection.time).label('hour'),
            func.count(Detection.id).label('count')
        ).filter(Detection.time >= start_date).group_by('hour').all()
        
        hourly_distribution = {int(hour): count for hour, count in hourly_query}
        
        # Daily distribution (last 7 days)
        daily_query = self.db.session.query(
            func.date(Detection.time).label('date'),
            func.count(Detection.id).label('count')
        ).filter(Detection.time >= week_start).group_by('date').all()
        
        daily_distribution = {str(date): count for date, count in daily_query}
        
        # Average confidence (if available in your data model)
        # This would need to be added to the Detection model
        avg_confidence = 0.85  # Placeholder
        
        # Active sensors (zones with recent activity)
        active_sensors = self.db.session.query(func.count(func.distinct(Detection.zone))).filter(
            Detection.time >= now - timedelta(hours=24),
            Detection.zone.isnot(None)
        ).scalar() or 0
        
        return AnalyticsMetrics(
            total_detections=total_detections,
            unique_species=unique_species,
            active_sensors=active_sensors,
            avg_confidence=avg_confidence,
            detections_today=detections_today,
            detections_this_week=detections_this_week,
            detections_this_month=detections_this_month,
            top_species=top_species,
            hourly_distribution=hourly_distribution,
            daily_distribution=daily_distribution
        )
    
    def get_species_insights(self, species: str, days: int = 30) -> Dict[str, Any]:
        """Get detailed insights for a specific species."""
        from web.app import Detection
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Basic species metrics
        detections = Detection.query.filter(
            Detection.species == species,
            Detection.time >= start_date
        ).all()
        
        if not detections:
            return {'error': f'No data found for species: {species}'}
        
        # Zone distribution
        zone_counts = Counter(d.zone for d in detections if d.zone)
        
        # Time patterns
        hour_counts = Counter(d.time.hour for d in detections)
        
        # Recent activity
        recent_detections = sorted(detections, key=lambda x: x.time, reverse=True)[:10]
        
        return {
            'species': species,
            'total_detections': len(detections),
            'zones': dict(zone_counts.most_common(10)),
            'hourly_pattern': dict(hour_counts),
            'recent_detections': [
                {
                    'id': d.id,
                    'time': d.time.isoformat(),
                    'zone': d.zone,
                    'latitude': d.latitude,
                    'longitude': d.longitude
                } for d in recent_detections
            ],
            'first_detection': min(detections, key=lambda x: x.time).time.isoformat(),
            'last_detection': max(detections, key=lambda x: x.time).time.isoformat()
        }
    
    def get_zone_analytics(self, zone: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a specific zone/sensor."""
        from web.app import Detection
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        detections = Detection.query.filter(
            Detection.zone == zone,
            Detection.time >= start_date
        ).all()
        
        if not detections:
            return {'error': f'No data found for zone: {zone}'}
        
        # Species diversity
        species_counts = Counter(d.species for d in detections)
        
        # Activity timeline
        daily_counts = defaultdict(int)
        for detection in detections:
            date_key = detection.time.date().isoformat()
            daily_counts[date_key] += 1
        
        return {
            'zone': zone,
            'total_detections': len(detections),
            'species_diversity': len(species_counts),
            'species_breakdown': dict(species_counts.most_common()),
            'daily_activity': dict(daily_counts),
            'avg_detections_per_day': len(detections) / days,
            'peak_activity_hour': Counter(d.time.hour for d in detections).most_common(1)[0] if detections else None
        }


class ChartGenerator:
    """Generate interactive charts for analytics dashboard."""
    
    def __init__(self):
        """Initialize chart generator."""
        self.config = get_config()
    
    def create_species_bar_chart(self, species_data: List[Tuple[str, int]]) -> str:
        """Create bar chart for top species."""
        if not PLOTLY_AVAILABLE:
            return self._create_fallback_chart("Species data unavailable - Plotly not installed")
        
        if not species_data:
            return self._create_fallback_chart("No species data available")
        
        species, counts = zip(*species_data)
        
        fig = go.Figure(data=[
            go.Bar(
                x=species,
                y=counts,
                marker_color='rgb(55, 83, 109)',
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Top Species Detections',
            xaxis_title='Species',
            yaxis_title='Number of Detections',
            template='plotly_white',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def create_hourly_activity_chart(self, hourly_data: Dict[int, int]) -> str:
        """Create line chart for hourly activity patterns."""
        if not PLOTLY_AVAILABLE:
            return self._create_fallback_chart("Hourly data unavailable - Plotly not installed")
        
        # Fill in missing hours with 0
        hours = list(range(24))
        counts = [hourly_data.get(hour, 0) for hour in hours]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=hours,
                y=counts,
                mode='lines+markers',
                line=dict(color='rgb(75, 192, 192)', width=3),
                marker=dict(size=6)
            )
        ])
        
        fig.update_layout(
            title='Wildlife Activity by Hour of Day',
            xaxis_title='Hour (24h format)',
            yaxis_title='Number of Detections',
            template='plotly_white',
            height=400,
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=2
            )
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def create_daily_trend_chart(self, daily_data: Dict[str, int]) -> str:
        """Create trend chart for daily detections."""
        if not PLOTLY_AVAILABLE:
            return self._create_fallback_chart("Daily data unavailable - Plotly not installed")
        
        if not daily_data:
            return self._create_fallback_chart("No daily data available")
        
        dates = sorted(daily_data.keys())
        counts = [daily_data[date] for date in dates]
        
        fig = go.Figure(data=[
            go.Scatter(
                x=dates,
                y=counts,
                mode='lines+markers',
                fill='tonexty',
                line=dict(color='rgb(128, 128, 255)', width=2),
                marker=dict(size=4)
            )
        ])
        
        fig.update_layout(
            title='Daily Detection Trends (Last 7 Days)',
            xaxis_title='Date',
            yaxis_title='Number of Detections',
            template='plotly_white',
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def create_zone_heatmap(self, zone_data: Dict[str, int]) -> str:
        """Create heatmap for zone activity."""
        if not PLOTLY_AVAILABLE or not zone_data:
            return self._create_fallback_chart("Zone data unavailable")
        
        zones = list(zone_data.keys())
        counts = list(zone_data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                y=zones,
                x=counts,
                orientation='h',
                marker_color='rgb(158, 185, 243)'
            )
        ])
        
        fig.update_layout(
            title='Detection Activity by Zone',
            xaxis_title='Number of Detections',
            yaxis_title='Zone',
            template='plotly_white',
            height=max(300, len(zones) * 40)
        )
        
        return fig.to_html(include_plotlyjs='cdn')
    
    def _create_fallback_chart(self, message: str) -> str:
        """Create fallback HTML when charts can't be generated."""
        return f"""
        <div style="
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            color: #666;
            background-color: #f9f9f9;
            height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div>
                <h3>Chart Unavailable</h3>
                <p>{message}</p>
            </div>
        </div>
        """


class ReportGenerator:
    """Generate PDF and CSV reports."""
    
    def __init__(self, analytics_engine: AnalyticsEngine):
        """Initialize report generator."""
        self.analytics_engine = analytics_engine
        self.config = get_config()
    
    def generate_csv_report(self, start_date: datetime, end_date: datetime) -> io.StringIO:
        """Generate CSV report for detections."""
        from web.app import Detection
        
        detections = Detection.query.filter(
            Detection.time >= start_date,
            Detection.time <= end_date
        ).order_by(Detection.time.desc()).all()
        
        # Create CSV data
        output = io.StringIO()
        output.write('ID,Species,Time,Zone,Latitude,Longitude,Image URL\\n')
        
        for detection in detections:
            output.write(f'{detection.id},{detection.species},{detection.time.isoformat()},')
            output.write(f'{detection.zone or ""},{detection.latitude or ""},{detection.longitude or ""},')
            output.write(f'{detection.image_url or ""}\\n')
        
        output.seek(0)
        return output
    
    def generate_pdf_report(self, metrics: AnalyticsMetrics, period_days: int = 30) -> io.BytesIO:
        """Generate PDF analytics report."""
        if not PDF_AVAILABLE:
            raise ImportError("PDF generation requires fpdf2: pip install fpdf2")
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        # Title
        pdf.cell(200, 10, 'NightScan Wildlife Analytics Report', ln=True, align='C')
        pdf.ln(10)
        
        # Report period
        pdf.set_font('Arial', '', 12)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        pdf.cell(200, 10, f'Report Period: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}', ln=True, align='C')
        pdf.ln(10)
        
        # Summary metrics
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(200, 10, 'Summary Metrics', ln=True)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        summary_data = [
            ('Total Detections', metrics.total_detections),
            ('Unique Species', metrics.unique_species),
            ('Active Sensors', metrics.active_sensors),
            ('Detections Today', metrics.detections_today),
            ('Detections This Week', metrics.detections_this_week),
            ('Detections This Month', metrics.detections_this_month),
        ]
        
        for label, value in summary_data:
            pdf.cell(100, 8, f'{label}:', 0, 0)
            pdf.cell(100, 8, str(value), 0, 1)
        
        pdf.ln(10)
        
        # Top species
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(200, 10, 'Top Species', ln=True)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 12)
        for i, (species, count) in enumerate(metrics.top_species[:10], 1):
            pdf.cell(200, 6, f'{i}. {species}: {count} detections', ln=True)
        
        # Convert to bytes
        output = io.BytesIO()
        pdf_string = pdf.output(dest='S').encode('latin-1')
        output.write(pdf_string)
        output.seek(0)
        
        return output


# Route handlers
@analytics_bp.route('/dashboard')
@login_required
@track_request_metrics
def dashboard():
    """Main analytics dashboard."""
    try:
        from web.app import db
        
        analytics_engine = AnalyticsEngine(db)
        chart_generator = ChartGenerator()
        
        # Get metrics for last 30 days
        days = int(request.args.get('days', 30))
        metrics = analytics_engine.get_detection_metrics(days)
        
        # Generate charts
        species_chart = chart_generator.create_species_bar_chart(metrics.top_species)
        hourly_chart = chart_generator.create_hourly_activity_chart(metrics.hourly_distribution)
        daily_chart = chart_generator.create_daily_trend_chart(metrics.daily_distribution)
        
        # Zone data (get from detections)
        from web.app import Detection
        zone_query = db.session.query(
            Detection.zone, func.count(Detection.id).label('count')
        ).filter(
            Detection.zone.isnot(None),
            Detection.time >= datetime.utcnow() - timedelta(days=days)
        ).group_by(Detection.zone).all()
        
        zone_data = {zone: count for zone, count in zone_query}
        zone_chart = chart_generator.create_zone_heatmap(zone_data)
        
        # Dashboard HTML template
        dashboard_html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>NightScan Analytics Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
                .chart-container { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            </style>
        </head>
        <body class="bg-light">
            <div class="container-fluid py-4">
                <div class="row mb-4">
                    <div class="col">
                        <h1 class="display-4">NightScan Analytics</h1>
                        <p class="lead">Wildlife detection insights for the last {{ days }} days</p>
                    </div>
                    <div class="col-auto">
                        <a href="/analytics/export/csv" class="btn btn-outline-primary">Export CSV</a>
                        <a href="/analytics/export/pdf" class="btn btn-primary">Export PDF</a>
                    </div>
                </div>
                
                <!-- Metrics Cards -->
                <div class="row mb-4">
                    <div class="col-md-2">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Total Detections</h5>
                                <h2 class="display-6">{{ metrics.total_detections }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Species</h5>
                                <h2 class="display-6">{{ metrics.unique_species }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Active Sensors</h5>
                                <h2 class="display-6">{{ metrics.active_sensors }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">Today</h5>
                                <h2 class="display-6">{{ metrics.detections_today }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">This Week</h5>
                                <h2 class="display-6">{{ metrics.detections_this_week }}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-2">
                        <div class="card metric-card">
                            <div class="card-body text-center">
                                <h5 class="card-title">This Month</h5>
                                <h2 class="display-6">{{ metrics.detections_this_month }}</h2>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Charts -->
                <div class="row mb-4">
                    <div class="col-lg-6">
                        <div class="card chart-container">
                            <div class="card-body">
                                {{ species_chart|safe }}
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-6">
                        <div class="card chart-container">
                            <div class="card-body">
                                {{ hourly_chart|safe }}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-4">
                    <div class="col-lg-6">
                        <div class="card chart-container">
                            <div class="card-body">
                                {{ daily_chart|safe }}
                            </div>
                        </div>
                    </div>
                    <div class="col-lg-6">
                        <div class="card chart-container">
                            <div class="card-body">
                                {{ zone_chart|safe }}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
        '''
        
        return render_template_string(
            dashboard_html,
            metrics=metrics,
            days=days,
            species_chart=species_chart,
            hourly_chart=hourly_chart,
            daily_chart=daily_chart,
            zone_chart=zone_chart
        )
        
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return jsonify({'error': 'Failed to load dashboard'}), 500


@analytics_bp.route('/api/metrics')
@login_required
@track_request_metrics
def api_metrics():
    """API endpoint for analytics metrics."""
    try:
        from web.app import db
        
        analytics_engine = AnalyticsEngine(db)
        days = int(request.args.get('days', 30))
        metrics = analytics_engine.get_detection_metrics(days)
        
        return jsonify(asdict(metrics))
        
    except Exception as e:
        logger.error(f"API metrics error: {e}")
        return jsonify({'error': 'Failed to get metrics'}), 500


@analytics_bp.route('/api/species/<species>')
@login_required
@track_request_metrics
def species_insights(species):
    """Get insights for a specific species."""
    try:
        from web.app import db
        
        analytics_engine = AnalyticsEngine(db)
        days = int(request.args.get('days', 30))
        insights = analytics_engine.get_species_insights(species, days)
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Species insights error: {e}")
        return jsonify({'error': 'Failed to get species insights'}), 500


@analytics_bp.route('/export/csv')
@login_required
@track_request_metrics
def export_csv():
    """Export detections data as CSV."""
    try:
        from web.app import db
        
        # Date range
        days = int(request.args.get('days', 30))
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        analytics_engine = AnalyticsEngine(db)
        report_generator = ReportGenerator(analytics_engine)
        
        csv_data = report_generator.generate_csv_report(start_date, end_date)
        
        # Create response
        output = io.BytesIO()
        output.write(csv_data.getvalue().encode('utf-8'))
        output.seek(0)
        
        filename = f'nightscan_detections_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        return jsonify({'error': 'Failed to export CSV'}), 500


@analytics_bp.route('/export/pdf')
@login_required
@track_request_metrics
def export_pdf():
    """Export analytics report as PDF."""
    try:
        from web.app import db
        
        days = int(request.args.get('days', 30))
        
        analytics_engine = AnalyticsEngine(db)
        report_generator = ReportGenerator(analytics_engine)
        
        metrics = analytics_engine.get_detection_metrics(days)
        pdf_data = report_generator.generate_pdf_report(metrics, days)
        
        filename = f'nightscan_analytics_{datetime.utcnow().strftime("%Y%m%d")}.pdf'
        
        return send_file(
            pdf_data,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
        
    except ImportError:
        return jsonify({'error': 'PDF generation not available. Install fpdf2.'}), 500
    except Exception as e:
        logger.error(f"PDF export error: {e}")
        return jsonify({'error': 'Failed to export PDF'}), 500


if __name__ == "__main__":
    # Test analytics functionality
    print("Analytics dashboard module loaded")
    if PLOTLY_AVAILABLE:
        print("Plotly available - charts enabled")
    else:
        print("Plotly not available - install with: pip install plotly pandas")
    
    if PDF_AVAILABLE:
        print("PDF generation available")
    else:
        print("PDF generation not available - install with: pip install fpdf2")