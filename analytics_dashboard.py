"""Optimized Analytics and reporting dashboard for NightScan - No N+1 queries."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Iterator
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

from flask import Blueprint, render_template_string, jsonify, request, send_file, Response
from flask_login import login_required, current_user
from sqlalchemy import func, and_, or_, desc, text
from datetime import datetime

from config import get_config
from metrics import track_request_metrics
from cache_manager import cache_analytics_result, invalidate_analytics_cache, get_cache_manager
from cache_middleware import cache_for_analytics

logger = logging.getLogger(__name__)

# Create analytics blueprint
analytics_bp = Blueprint('analytics', __name__, url_prefix='/analytics')

# Constants for performance
MAX_EXPORT_ROWS = 10000  # Maximum rows for CSV export
PAGINATION_SIZE = 1000   # Rows per page for exports
CACHE_TTL = 300         # 5 minutes cache


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


class OptimizedAnalyticsEngine:
    """Optimized analytics engine with no N+1 queries."""
    
    def __init__(self, db):
        """Initialize analytics engine with database connection."""
        self.db = db
        self.config = get_config()
    
    @cache_analytics_result(ttl=300)  # Cache for 5 minutes
    def get_detection_metrics(self, days: int = 30) -> AnalyticsMetrics:
        """Get comprehensive detection metrics using aggregated queries."""
        from web.app import Detection
        
        # Date ranges
        now = datetime.utcnow()
        start_date = now - timedelta(days=days)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = now - timedelta(days=7)
        month_start = now - timedelta(days=30)
        
        # Single aggregated query for all basic metrics
        metrics_query = self.db.session.query(
            func.count(Detection.id).label('total'),
            func.count(func.distinct(Detection.species)).label('unique_species'),
            func.avg(Detection.confidence).label('avg_confidence'),
            func.sum(
                func.cast(Detection.time >= today_start, db.Integer)
            ).label('today'),
            func.sum(
                func.cast(Detection.time >= week_start, db.Integer)
            ).label('week'),
            func.sum(
                func.cast(Detection.time >= month_start, db.Integer)
            ).label('month')
        ).filter(Detection.time >= start_date).first()
        
        # Active sensors (single query)
        active_sensors = self.db.session.query(
            func.count(func.distinct(Detection.zone))
        ).filter(
            Detection.time >= now - timedelta(hours=24),
            Detection.zone.isnot(None)
        ).scalar() or 0
        
        # Top species (already optimized)
        top_species = self.db.session.query(
            Detection.species, 
            func.count(Detection.id).label('count')
        ).filter(
            Detection.time >= start_date
        ).group_by(
            Detection.species
        ).order_by(
            desc('count')
        ).limit(10).all()
        
        # Hourly distribution (optimized)
        hourly_dist = self.db.session.query(
            func.extract('hour', Detection.time).label('hour'),
            func.count(Detection.id).label('count')
        ).filter(
            Detection.time >= start_date
        ).group_by('hour').all()
        
        # Daily distribution (optimized)
        daily_dist = self.db.session.query(
            func.date(Detection.time).label('date'),
            func.count(Detection.id).label('count')
        ).filter(
            Detection.time >= week_start
        ).group_by('date').all()
        
        return AnalyticsMetrics(
            total_detections=metrics_query.total or 0,
            unique_species=metrics_query.unique_species or 0,
            active_sensors=active_sensors,
            avg_confidence=float(metrics_query.avg_confidence or 0),
            detections_today=metrics_query.today or 0,
            detections_this_week=metrics_query.week or 0,
            detections_this_month=metrics_query.month or 0,
            top_species=[(s, c) for s, c in top_species],
            hourly_distribution={int(h): c for h, c in hourly_dist},
            daily_distribution={str(d): c for d, c in daily_dist}
        )
    
    @cache_analytics_result(ttl=300)  # Cache for 5 minutes
    def get_species_insights_optimized(self, species: str, days: int = 30) -> Dict[str, Any]:
        """Get detailed insights for a specific species without loading all records."""
        from web.app import Detection
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Check if species exists and get count
        species_stats = self.db.session.query(
            func.count(Detection.id).label('total'),
            func.min(Detection.time).label('first'),
            func.max(Detection.time).label('last')
        ).filter(
            Detection.species == species,
            Detection.time >= start_date
        ).first()
        
        if not species_stats.total:
            return {'error': f'No data found for species: {species}'}
        
        # Zone distribution (aggregated query)
        zone_dist = self.db.session.query(
            Detection.zone,
            func.count(Detection.id).label('count')
        ).filter(
            Detection.species == species,
            Detection.time >= start_date,
            Detection.zone.isnot(None)
        ).group_by(
            Detection.zone
        ).order_by(
            desc('count')
        ).limit(10).all()
        
        # Hourly pattern (aggregated query)
        hourly_pattern = self.db.session.query(
            func.extract('hour', Detection.time).label('hour'),
            func.count(Detection.id).label('count')
        ).filter(
            Detection.species == species,
            Detection.time >= start_date
        ).group_by('hour').all()
        
        # Recent detections (limited to 10)
        recent = self.db.session.query(Detection).filter(
            Detection.species == species,
            Detection.time >= start_date
        ).order_by(
            Detection.time.desc()
        ).limit(10).all()
        
        return {
            'species': species,
            'total_detections': species_stats.total,
            'zones': {z: c for z, c in zone_dist},
            'hourly_pattern': {int(h): c for h, c in hourly_pattern},
            'recent_detections': [
                {
                    'id': d.id,
                    'time': d.time.isoformat(),
                    'zone': d.zone,
                    'latitude': d.latitude,
                    'longitude': d.longitude
                } for d in recent
            ],
            'first_detection': species_stats.first.isoformat() if species_stats.first else None,
            'last_detection': species_stats.last.isoformat() if species_stats.last else None
        }
    
    @cache_analytics_result(ttl=300)  # Cache for 5 minutes
    def get_zone_analytics_optimized(self, zone: str, days: int = 30) -> Dict[str, Any]:
        """Get analytics for a specific zone without loading all records."""
        from web.app import Detection
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Basic zone stats
        zone_stats = self.db.session.query(
            func.count(Detection.id).label('total'),
            func.count(func.distinct(Detection.species)).label('species_count')
        ).filter(
            Detection.zone == zone,
            Detection.time >= start_date
        ).first()
        
        if not zone_stats.total:
            return {'error': f'No data found for zone: {zone}'}
        
        # Species breakdown (aggregated)
        species_breakdown = self.db.session.query(
            Detection.species,
            func.count(Detection.id).label('count')
        ).filter(
            Detection.zone == zone,
            Detection.time >= start_date
        ).group_by(
            Detection.species
        ).order_by(
            desc('count')
        ).all()
        
        # Daily activity (aggregated)
        daily_activity = self.db.session.query(
            func.date(Detection.time).label('date'),
            func.count(Detection.id).label('count')
        ).filter(
            Detection.zone == zone,
            Detection.time >= start_date
        ).group_by('date').all()
        
        # Peak hour calculation
        peak_hour = self.db.session.query(
            func.extract('hour', Detection.time).label('hour'),
            func.count(Detection.id).label('count')
        ).filter(
            Detection.zone == zone,
            Detection.time >= start_date
        ).group_by('hour').order_by(desc('count')).first()
        
        return {
            'zone': zone,
            'total_detections': zone_stats.total,
            'species_diversity': zone_stats.species_count,
            'species_breakdown': {s: c for s, c in species_breakdown},
            'daily_activity': {str(d): c for d, c in daily_activity},
            'avg_detections_per_day': zone_stats.total / days,
            'peak_activity_hour': (int(peak_hour.hour), peak_hour.count) if peak_hour else None
        }


class OptimizedReportGenerator:
    """Generate reports with pagination to avoid memory issues."""
    
    def __init__(self, analytics_engine: OptimizedAnalyticsEngine):
        """Initialize report generator."""
        self.analytics_engine = analytics_engine
        self.config = get_config()
    
    def generate_csv_report_paginated(self, start_date: datetime, end_date: datetime) -> Iterator[str]:
        """Generate CSV report using pagination to avoid loading all data."""
        from web.app import Detection
        
        # Yield header
        yield 'ID,Species,Time,Zone,Latitude,Longitude,Image URL\n'
        
        # Get total count first
        total_count = self.db.session.query(
            func.count(Detection.id)
        ).filter(
            Detection.time >= start_date,
            Detection.time <= end_date
        ).scalar()
        
        if total_count > MAX_EXPORT_ROWS:
            logger.warning(f"Export limited to {MAX_EXPORT_ROWS} rows out of {total_count}")
        
        # Paginate through results
        page = 1
        exported = 0
        
        while exported < min(total_count, MAX_EXPORT_ROWS):
            # Get page of results
            detections = Detection.query.filter(
                Detection.time >= start_date,
                Detection.time <= end_date
            ).order_by(
                Detection.time.desc()
            ).paginate(
                page=page, 
                per_page=PAGINATION_SIZE,
                error_out=False
            )
            
            if not detections.items:
                break
            
            # Yield rows
            for detection in detections.items:
                yield (
                    f'{detection.id},{detection.species},'
                    f'{detection.time.isoformat()},'
                    f'{detection.zone or ""},{detection.latitude or ""},'
                    f'{detection.longitude or ""},{detection.image_url or ""}\n'
                )
                exported += 1
                
                if exported >= MAX_EXPORT_ROWS:
                    break
            
            page += 1
    
    def generate_streaming_csv_response(self, start_date: datetime, end_date: datetime) -> Response:
        """Generate streaming CSV response."""
        def generate():
            for row in self.generate_csv_report_paginated(start_date, end_date):
                yield row
        
        return Response(
            generate(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=nightscan_detections_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
            }
        )
    
    def generate_pdf_report(self, metrics: AnalyticsMetrics, period_days: int = 30) -> io.BytesIO:
        """Generate PDF analytics report (unchanged as it uses pre-aggregated data)."""
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
            ('Average Confidence', f'{metrics.avg_confidence:.2%}'),
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


# Unchanged chart generator (works with aggregated data)
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


# Optimized route handlers
@analytics_bp.route('/dashboard')
@login_required
@track_request_metrics
def dashboard():
    """Main analytics dashboard with optimized queries."""
    try:
        from web.app import db, Detection
        
        analytics_engine = OptimizedAnalyticsEngine(db)
        chart_generator = ChartGenerator()
        
        # Get metrics for last 30 days
        days = int(request.args.get('days', 30))
        
        # Validate days parameter
        if days > 365:
            days = 365  # Limit to 1 year
        
        metrics = analytics_engine.get_detection_metrics(days)
        
        # Generate charts
        species_chart = chart_generator.create_species_bar_chart(metrics.top_species)
        hourly_chart = chart_generator.create_hourly_activity_chart(metrics.hourly_distribution)
        daily_chart = chart_generator.create_daily_trend_chart(metrics.daily_distribution)
        
        # Zone data (already optimized query)
        zone_query = db.session.query(
            Detection.zone, 
            func.count(Detection.id).label('count')
        ).filter(
            Detection.zone.isnot(None),
            Detection.time >= datetime.utcnow() - timedelta(days=days)
        ).group_by(
            Detection.zone
        ).order_by(
            desc('count')
        ).limit(20).all()  # Limit zones to top 20
        
        zone_data = {zone: count for zone, count in zone_query}
        zone_chart = chart_generator.create_zone_heatmap(zone_data)
        
        # Dashboard HTML template (unchanged)
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
                        <a href="/analytics/export/csv?days={{ days }}" class="btn btn-outline-primary">Export CSV</a>
                        <a href="/analytics/export/pdf?days={{ days }}" class="btn btn-primary">Export PDF</a>
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
                
                <div class="row">
                    <div class="col-12 text-center text-muted">
                        <small>Performance optimized - Large datasets are paginated</small>
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
@cache_for_analytics(ttl=300)  # Cache for 5 minutes
def api_metrics():
    """API endpoint for analytics metrics."""
    try:
        from web.app import db
        
        analytics_engine = OptimizedAnalyticsEngine(db)
        days = int(request.args.get('days', 30))
        
        # Limit days
        days = min(days, 365)
        
        metrics = analytics_engine.get_detection_metrics(days)
        
        return jsonify(asdict(metrics))
        
    except Exception as e:
        logger.error(f"API metrics error: {e}")
        return jsonify({'error': 'Failed to get metrics'}), 500


@analytics_bp.route('/api/species/<species>')
@login_required
@track_request_metrics
@cache_for_analytics(ttl=300)  # Cache for 5 minutes
def species_insights(species):
    """Get insights for a specific species (optimized)."""
    try:
        from web.app import db
        
        analytics_engine = OptimizedAnalyticsEngine(db)
        days = int(request.args.get('days', 30))
        
        # Limit days
        days = min(days, 365)
        
        insights = analytics_engine.get_species_insights_optimized(species, days)
        
        return jsonify(insights)
        
    except Exception as e:
        logger.error(f"Species insights error: {e}")
        return jsonify({'error': 'Failed to get species insights'}), 500


@analytics_bp.route('/api/zone/<zone>')
@login_required
@track_request_metrics
@cache_for_analytics(ttl=300)  # Cache for 5 minutes
def zone_analytics(zone):
    """Get analytics for a specific zone (optimized)."""
    try:
        from web.app import db
        
        analytics_engine = OptimizedAnalyticsEngine(db)
        days = int(request.args.get('days', 30))
        
        # Limit days
        days = min(days, 365)
        
        analytics = analytics_engine.get_zone_analytics_optimized(zone, days)
        
        return jsonify(analytics)
        
    except Exception as e:
        logger.error(f"Zone analytics error: {e}")
        return jsonify({'error': 'Failed to get zone analytics'}), 500


@analytics_bp.route('/export/csv')
@login_required
@track_request_metrics
def export_csv():
    """Export detections data as CSV with streaming."""
    try:
        from web.app import db
        
        # Date range
        days = int(request.args.get('days', 30))
        days = min(days, 365)  # Limit to 1 year
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        analytics_engine = OptimizedAnalyticsEngine(db)
        report_generator = OptimizedReportGenerator(analytics_engine)
        
        # Return streaming response
        return report_generator.generate_streaming_csv_response(start_date, end_date)
        
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
        days = min(days, 365)  # Limit to 1 year
        
        analytics_engine = OptimizedAnalyticsEngine(db)
        report_generator = OptimizedReportGenerator(analytics_engine)
        
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
    print("Optimized analytics dashboard module loaded")
    print("Key optimizations:")
    print("- No N+1 queries - all data aggregated in SQL")
    print("- Streaming CSV export with pagination")
    print("- Limited result sets (max 10k rows for export)")
    print("- Query result limits (top 20 zones, last 365 days max)")
    
    if PLOTLY_AVAILABLE:
        print("✓ Plotly available - charts enabled")
    else:
        print("✗ Plotly not available - install with: pip install plotly pandas")
    
    if PDF_AVAILABLE:
        print("✓ PDF generation available")
    else:
        print("✗ PDF generation not available - install with: pip install fpdf2")