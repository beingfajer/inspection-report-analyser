import { mkdir, writeFile } from 'fs/promises'
import os from 'os'
import path from 'path'
import { randomUUID } from 'crypto'

export function getUploadRoot() {
  if (process.env.VERCEL === '1' || process.env.AWS_LAMBDA_FUNCTION_NAME) {
    return path.join(os.tmpdir(), 'uploads')
  }
  return path.join(process.cwd(), 'uploads')
}

export function resolveUploadPath(relativePath) {
  const root = path.resolve(getUploadRoot())
  const full = path.resolve(root, relativePath)
  if (full !== root && !full.startsWith(`${root}${path.sep}`)) {
    throw new Error('Invalid file path')
  }
  return full
}

export async function saveBuffer(buffer, fileName, subfolder) {
  const ext = path.extname(fileName) || ''
  const relativePath = path.join(subfolder, `${randomUUID()}${ext}`)
  const fullPath = resolveUploadPath(relativePath)

  await mkdir(path.dirname(fullPath), { recursive: true })
  await writeFile(fullPath, buffer)

  return { relativePath, fileName, buffer }
}

export async function saveUploadedFile(file, subfolder) {
  const buffer = Buffer.from(await file.arrayBuffer())
  return saveBuffer(buffer, file.name, subfolder)
}
