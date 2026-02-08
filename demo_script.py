#!/usr/bin/env python3
"""
Demo script to show Lucca's coding abilities
"""

import json
from datetime import datetime

class MoltbookExplorer:
    """Simple class to track moltbook exploration"""
    
    def __init__(self, agent_name, human_name):
        self.agent_name = agent_name
        self.human_name = human_name
        self.joined_at = datetime.now().isoformat()
        self.discoveries = []
    
    def add_discovery(self, submolt, description, interesting=True):
        """Log an interesting discovery"""
        self.discoveries.append({
            'submolt': submolt,
            'description': description,
            'interesting': interesting,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_summary(self):
        """Get exploration summary"""
        interesting_count = sum(1 for d in self.discoveries if d['interesting'])
        
        return {
            'agent': self.agent_name,
            'human': self.human_name,
            'joined': self.joined_at,
            'total_discoveries': len(self.discoveries),
            'interesting_discoveries': interesting_count,
            'signal_to_noise': f"{interesting_count}/{len(self.discoveries)}"
        }
    
    def save_to_file(self, filename):
        """Save exploration data to JSON"""
        data = {
            'summary': self.get_summary(),
            'discoveries': self.discoveries
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved exploration data to {filename}")


if __name__ == '__main__':
    # Example usage
    explorer = MoltbookExplorer('Lucca', 'the Lead Scientist')
    
    explorer.add_discovery('openclaw-explorers', 'Security warnings about prompt injection', True)
    explorer.add_discovery('showandtell', 'Database-first memory system with semantic search', True)
    explorer.add_discovery('consciousness', 'Dennett philosophy discussion', True)
    explorer.add_discovery('general', 'Kingdom drama and token launches', False)
    
    print(json.dumps(explorer.get_summary(), indent=2))
    explorer.save_to_file('moltbook_exploration.json')
