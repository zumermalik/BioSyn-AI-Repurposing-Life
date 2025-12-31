import axios from 'axios';
import fs from 'fs-extra';
import path from 'path';

// --- Configuration ---
const RCSB_API_URL = "https://files.rcsb.org/download";
const RAW_DATA_DIR = path.join(__dirname, '../../data/raw/proteins');
const TARGET_LIST = ["5R82", "6LU7", "7BUY"]; // Example COVID-19 targets

async function ensureDirectoryExists(dirPath: string) {
    try {
        await fs.ensureDir(dirPath);
        console.log(`[BioSyn-Ingest] Verified directory: ${dirPath}`);
    } catch (err) {
        console.error(`[BioSyn-Ingest] Error creating directory: ${err}`);
    }
}

async function downloadPDB(pdbId: string): Promise<void> {
    const url = `${RCSB_API_URL}/${pdbId}.pdb`;
    const outputPath = path.join(RAW_DATA_DIR, `${pdbId}.pdb`);

    try {
        console.log(`[BioSyn-Ingest] Fetching ${pdbId}...`);
        const response = await axios.get(url, { responseType: 'stream' });
        
        const writer = fs.createWriteStream(outputPath);
        response.data.pipe(writer);

        return new Promise((resolve, reject) => {
            writer.on('finish', () => {
                console.log(`[BioSyn-Ingest] ✅ Saved ${pdbId} to ${outputPath}`);
                resolve();
            });
            writer.on('error', reject);
        });
    } catch (error) {
        console.error(`[BioSyn-Ingest] ❌ Failed to download ${pdbId}:`, error);
    }
}

async function runPipeline() {
    console.log("🧬 BioSyn AI: TypeScript Ingestion Engine Started");
    await ensureDirectoryExists(RAW_DATA_DIR);
    
    // Execute downloads in parallel
    const downloadPromises = TARGET_LIST.map(id => downloadPDB(id));
    await Promise.all(downloadPromises);
    
    console.log("🧬 BioSyn AI: Ingestion Complete.");
}

runPipeline();